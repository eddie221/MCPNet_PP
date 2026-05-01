import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
import torch
import os
import random
import numpy as np
import tqdm
from typing import Callable, Dict, List, Optional, Tuple
from train_utils import prepare_transforms, main_process_first

def find_classes(directory: str, num_class: int) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    np.random.seed(0)
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    random_pick_class = np.random.choice(classes, num_class, replace = False)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(random_pick_class)}
    return classes, class_to_idx

class ImageNet_subset(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        num_class: int,
        transform: Optional[Callable] = None,
    ):
        self.num_class = num_class
        super().__init__(root, transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        
        return find_classes(directory, self.num_class)

class RandomSampleDataset(Dataset):
    def __init__(self, dataset, num_samples_per_class):
        self.dataset = dataset
        self.num_samples_per_class = num_samples_per_class
        self.indices_per_class = self._get_indices_per_class()
        self.indices = self._sample_indices()
        # print("finish")

    def _get_indices_per_class(self):
        class_indices = {class_idx: [] for class_idx in range(len(self.dataset.classes))}
    
        # Gather indices for each class
        for idx, (path, class_idx) in tqdm.tqdm(enumerate(self.dataset.samples), total = len(self.dataset)):
            class_indices[class_idx].append(idx)

        return class_indices

    def _sample_indices(self):
        sampled_indices = []
        for indices in self.indices_per_class.values():
            if len(indices) >= self.num_samples_per_class:
                sampled_indices.extend(np.random.choice(indices, self.num_samples_per_class, replace=False))
            else:
                sampled_indices.extend(indices)
        return sampled_indices

    def resample(self):
        self.indices = self._sample_indices()

    def __getitem__(self, index):
        orig_index = self.indices[index]
        return self.dataset[orig_index]

    def __len__(self):
        return len(self.indices)

def get_random_sample_indices(dataset, num_samples_per_class):
    class_indices = {class_idx: [] for class_idx in range(len(dataset.classes))}
    
    # Gather indices for each class
    for idx, (path, class_idx) in enumerate(dataset.samples):
        class_indices[class_idx].append(idx)
    
    selected_indices = []
    # Randomly select 'num_samples_per_class' indices for each class
    for label, indices in zip(class_indices.keys(), class_indices.values()):
        if len(indices) >= num_samples_per_class:
            selected_indices.extend(random.sample(indices, num_samples_per_class))
        else:
            # If a class has fewer samples than 'num_samples_per_class', take all from that class
            print(f"Label : {label} has less sample than num_samples_per_class. Total : {len(indices)} select : {num_samples_per_class} !!")
            selected_indices.extend(indices)
    
    return selected_indices

def create_dataloader(path, batch_size, shuffle, n_workers, rank, mode, args):
    with main_process_first(rank):
        data_transforms = prepare_transforms(args)
        if rank in [-1, 0]:
            print("{} data_transforms : ".format(mode))
            print(data_transforms)
        dataset = torchvision.datasets.ImageFolder(path, data_transforms[mode])

    torch.distributed.barrier()

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle) if rank != -1 else None
    num_workers = min([os.cpu_count() // args.world_size, batch_size if batch_size > 1 else 0, n_workers])  # number of workers
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        sampler = sampler if rank != -1 else None,
                        shuffle = shuffle if rank == -1 else None,
                        num_workers = num_workers,
                        pin_memory = True) 

    return loader, dataset 

def create_dataloader_sub(dataset, select_idx, batch_size, shuffle, n_workers, rank, mode, args):
    with main_process_first(rank):
        data_transforms = prepare_transforms(args)
        if rank in [-1, 0]:
            print("{} data_transforms : ".format(mode))
            print(data_transforms)
        if select_idx is None:
            dataset = RandomSampleDataset(dataset, args.num_samples_per_class)
        else:
            dataset = Subset(dataset, select_idx)

    torch.distributed.barrier()

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle) if rank != -1 else None
    num_workers = min([os.cpu_count() // args.world_size, batch_size if batch_size > 1 else 0, n_workers])  # number of workers
    # print("num_workers : ", num_workers)
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        sampler = sampler if rank != -1 else None,
                        shuffle = shuffle if rank == -1 else None,
                        num_workers = num_workers,
                        pin_memory = True) 

    return loader, dataset 


def load_data(args):
    trainloader, traindataset = create_dataloader(args.train_dataset_path, 
                                                args.train_batch_size, 
                                                True,
                                                args.train_num_workers,
                                                args.global_rank,
                                                "train", args)

    # Define how many samples you want from each class
    num_samples_per_class = args.num_samples_per_class  # For example, 5 images per class

    if args.rand_sub:
        selected_indices = None
    else:
        selected_indices = get_random_sample_indices(traindataset, num_samples_per_class)
    sub_trainloader, sub_traindataset = create_dataloader_sub(traindataset, 
                                                      selected_indices, 
                                                      args.val_batch_size, 
                                                      True, 
                                                      args.val_num_workers,
                                                      args.global_rank, 
                                                      "val", 
                                                      args)

    valloader, valdataset = create_dataloader(args.val_dataset_path, 
                                            args.val_batch_size, 
                                            False,
                                            args.val_num_workers,
                                            args.global_rank, 
                                            "val", args)
    
    # combine
    dataloader = {"train" : trainloader, "val" : valloader, "sub_train" : sub_trainloader}
    dataset_sizes = {"train" : len(trainloader), 
                     "val" : len(valloader) if valloader is not None else 0, 
                     "sub_train" : len(sub_trainloader) if sub_trainloader is not None else 0}
    return dataloader, dataset_sizes, None