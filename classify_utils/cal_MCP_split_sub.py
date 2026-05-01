import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision
import numpy as np
import tqdm
import os
import time
import argparse
import sys
import random
from utils import get_layer_img_size, get_dataset, load_model, load_concept, load_weight
from train_utils import basic_args, model_args

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

def cal_MCP(model, concept_vecs, concept_means, data_transforms, data_path, args):
    print("Calculate MCP features")
    dataset = torchvision.datasets.ImageFolder(data_path, transform = data_transforms)
    Class_N = len(dataset.classes)
        
    N = len(dataset)
    print("Number of classes : ", Class_N)
    print("Number of images : ", len(dataset))

    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers = 16)

    # select the node
    concept_num = [concept_vecs[layer_i].shape[0] for layer_i in range(len(concept_vecs))]
    MCP_feats_raw = [[] for i in range(len(concept_num))]
    labels = []
    
    with torch.no_grad():
        for iteration, (img, label) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
            img = img.cuda()
            outputs = model(img)
            feats = outputs[0]
            labels.append(label)
            for layer_i, feat in enumerate(feats):
                if len(feat.shape) == 3:
                    feat = feat[:, 1:].permute(0, 2, 1)
                else:
                    feat = feat.flatten(2)

                B, D, N = feat.shape
                feat = feat.reshape(B, args.concept_per_layer[layer_i], args.concept_cha[layer_i], N)
                feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
                
                # calculate concept vector from covariance matrix
                concept_vector = concept_vecs[layer_i].cuda()
                response = torch.sum(feat * concept_vector.unsqueeze(0).unsqueeze(3), dim = 2)
                response = torch.nn.functional.adaptive_max_pool1d(response, 1)[..., 0]
                MCP_feats_raw[layer_i].append(response)
    
    for layer_i in range(len(MCP_feats_raw)):
        MCP_feats_raw[layer_i] = torch.cat(MCP_feats_raw[layer_i], dim = 0)
    MCP_feats_raw = torch.cat(MCP_feats_raw, dim = -1)
    labels = torch.cat(labels, dim = 0)

    torch.save(labels, f"{args.save_path}/labels.pkl")
    torch.save(MCP_feats_raw, f"{args.save_path}/MCP_features_raw.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument('--val_MCP', default = False, action = "store_true")
    parser.add_argument('--eigen_topk', default = 1, type = int)
    parser.add_argument("--device", default = "0", type = str)
    parser.add_argument("--param_root", type = str, required = True)
    args = parser.parse_args()
    print("Calculate the MCP features !!")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    case_name = args.case_name
    layer_sizes, image_size = get_layer_img_size(args.basic_model)
    print(args)
    data_path, train_path, val_path, num_classes = get_dataset(case_name)
    print(data_path)

    # load model
    extra_args = {}
    if "MCPNet_pp" in args.model:
        extra_args["sel_layers"] = args.sel_layers
        model = load_model(model = args.model, basic_model = args.basic_model, num_classes = num_classes, \
                        concept_per_layer = args.concept_per_layer, concept_cha = args.concept_cha, **extra_args)
    else:
        model = load_model(model = args.model, basic_model = args.basic_model, num_classes = num_classes, **extra_args)

    trained_param_path = f"{args.param_root}/{args.case_name}/{args.basic_model.lower()}/best_model.pkl"
    load_weight(model, trained_param_path)
    model.eval()

    args.save_path = f"./MCP_feats_tmp/{case_name}/{args.model.lower()}_{args.basic_model.lower()}"
    os.makedirs(args.save_path, exist_ok = True)

    # Load concept vectors and means
    concept_covs = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_covs"]
    concept_means = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_means"]
    concept_vecs, concept_means = load_concept(concept_covs, concept_means, args.eigen_topk)
            
    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    start = time.time()
    if args.val_MCP:
        cal_MCP(model, concept_vecs, concept_means, data_transforms, data_path + val_path, args)
    else:
        cal_MCP(model, concept_vecs, concept_means, data_transforms, data_path + train_path, args)
    print("Times : {} hrs".format((time.time() - start) / 3600))

