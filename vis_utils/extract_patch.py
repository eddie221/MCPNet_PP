import torch
import torchvision.transforms as transforms
import torchvision
import argparse
import os
import sys
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import tqdm
from utils import get_dataset, get_layer_img_size
from train_utils import basic_args, model_args

class customDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super(customDataset, self).__init__(root, transform)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img_name = image_path.split("\\")[-1]
        ori_img = Image.open(image_path).convert('RGB')
        W, H = ori_img.size

        img = self.transform(ori_img)
        
        return img, label, image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument('--scale', default = 1., type = float)
    parser.add_argument("--device", default = "0", type = str)
    parser.add_argument('--topk', type = int, choices = [25, 100, 225], required = True)
    parser.add_argument('--thumbnail', default = False, action = "store_true")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.topk == 25:
        result_w = 5
        result_h = 5
    elif args.topk == 100:
        result_w = 10
        result_h = 10
    else:
        result_w = 15
        result_h = 15

    image_size = 0
    layer_sizes, image_size = get_layer_img_size(args.basic_model)
    patch_sizes = []
    for i in range(len(layer_sizes)):
        patch_sizes.append(int(image_size // layer_sizes[i] * args.scale))

    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                          transforms.CenterCrop((image_size, image_size)),
                                          transforms.ToTensor(),
                                          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    

    # if args.thumbnail:
    #     save_dir = f"./extract_patch_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model}/"
    # else:
    #     save_dir = f"./extract_patch_tmp_split/{args.case_name}/{args.model.lower()}_{args.basic_model}/"
    save_dir = f"./extract_patch_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model}/"
    os.makedirs(save_dir, exist_ok = True)
    os.makedirs(f"{save_dir}/thumbnail", exist_ok = True)

    data_path, train_path, val_path, num_class = get_dataset(args.case_name)

    max_resp_index = torch.load(f"./find_topk_response_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model}/max_resp_value_idx.pkl", map_location = f"cuda:0")
    max_resp_value = torch.load(f"./find_topk_response_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model}/max_resp_value.pkl", map_location = f"cuda:0")

    train_dataset = customDataset(os.path.join(data_path, train_path), data_transforms)
    prototype_patches = []
    for layer_i, (index, f_size, patch_size) in enumerate(zip(max_resp_index, layer_sizes, patch_sizes)):
        image_id = (index // (f_size * f_size)).type(torch.int)
        pixel_place = (index % (f_size * f_size))
        for concept_i in tqdm.tqdm(range(args.concept_per_layer[layer_i])):
            all_patchs = np.zeros([result_h * patch_size * 2, result_w * patch_size * 2, 3])
            os.makedirs(f"{save_dir}/l{layer_i + 1}_{concept_i + 1}", exist_ok = True)
            # if args.thumbnail:
            #     all_patchs = np.zeros([result_h * patch_size * 2, result_w * patch_size * 2, 3])
            # else:
            #     os.makedirs(f"{save_dir}/l{layer_i + 1}_{concept_i + 1}", exist_ok = True)

            selected_imgs = []
            top_i = 0
            while len(selected_imgs) < args.topk:
                while image_id[concept_i, top_i] in selected_imgs:
                    top_i += 1
                    if top_i >= image_id.shape[1]:
                        break
                    
                selected_imgs.append(image_id[concept_i, top_i])
                stored_i = len(selected_imgs) - 1
                ori_img, label, img_path = train_dataset[image_id[concept_i, top_i]]

                ori_img = torch.nn.functional.interpolate(ori_img.unsqueeze(0), scale_factor = args.scale, mode = "bilinear")[0]
                ori_img = np.pad(ori_img.permute(1, 2, 0), ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), mode = "constant", constant_values = ((0, 0), (0, 0), (0, 0)))
                x_coor = ((pixel_place[concept_i, top_i] % f_size)).type(torch.int)
                y_coor = ((pixel_place[concept_i, top_i] // f_size)).type(torch.int)
                patch = ori_img[(y_coor - 1) * patch_size + patch_size: (y_coor + 1) * patch_size + patch_size, (x_coor - 1) * patch_size + patch_size : (x_coor + 1) * patch_size + patch_size, :] * 255

                # for thumbnail
                all_patchs[(stored_i // result_h) * patch_size * 2 : (stored_i // result_h + 1) * patch_size * 2, (stored_i % result_w) * patch_size * 2 : (stored_i % result_w + 1) * patch_size * 2] = patch
                # for individual
                patch[patch < 0] = 0
                result = Image.fromarray(patch.astype(np.uint8))
                result.save(f"{save_dir}/l{layer_i + 1}_{concept_i + 1}/l{layer_i + 1}_{concept_i + 1}_{stored_i + 1}.png")
                # if args.thumbnail:
                #     all_patchs[(stored_i // result_h) * patch_size * 2 : (stored_i // result_h + 1) * patch_size * 2, (stored_i % result_w) * patch_size * 2 : (stored_i % result_w + 1) * patch_size * 2] = patch
                # else:
                #     patch[patch < 0] = 0
                #     result = Image.fromarray(patch.astype(np.uint8))
                #     result.save(f"{save_dir}/l{layer_i + 1}_{concept_i + 1}/l{layer_i + 1}_{concept_i + 1}_{stored_i + 1}.png")

            result = Image.fromarray(all_patchs.astype(np.uint8))
            result.save(f"{save_dir}/thumbnail/l{layer_i + 1}_{concept_i + 1}.png")
            prototype_patches.append(all_patchs)

            # if args.thumbnail:
            #     result = Image.fromarray(all_patchs.astype(np.uint8))
            #     result.save(f"{save_dir}/l{layer_i + 1}_{concept_i + 1}.png")
            #     prototype_patches.append(all_patchs)

    prototype_patches = np.asarray(prototype_patches, dtype="object")
    np.save(f"{save_dir}/all_patches.npy", prototype_patches)