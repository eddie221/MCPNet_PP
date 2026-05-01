import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import argparse
import os
import numpy as np
from PIL import Image
import tqdm
import cv2
import matplotlib.pyplot as plt
from utils import load_concept, load_model, get_dataset, get_layer_img_size, load_weight
from train_utils import basic_args, model_args

class customDataset(ImageFolder):
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
        for trans in self.transform.transforms[:-1]:
            ori_img = trans(ori_img)
        
        return img, label, ori_img, image_path

def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def save_heatmap(ori_imgs, masks, args, layer_i = -1, concept_i = -1):
    heatmaps = []
    for img_i in range(args.topk):
        mask = masks[img_i, 0]
        mask -= mask.min()
        mask /= mask.max()
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.cpu()), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1] # OpenCV's BGR to RGB
        heatmap_img =  0.3 * np.float32(heatmap) + 0.7 * np.float32(ori_imgs[img_i].cpu().numpy().transpose(1,2,0))
        heatmaps.append(heatmap_img)
        if args.individually and layer_i != -1 and concept_i != -1:
            plt.imsave(fname = f"{args.saved_dir}/ori_img/l{layer_i + 1}_{concept_i + 1}_{img_i + 1}_heatmap.png", arr = heatmap_img, vmin = 0.0, vmax = 1.0)

    heatmaps = np.concatenate(heatmaps, axis = 1)
    return heatmaps

# red mask on the masked part
def save_masked(ori_imgs, masks, args):
    masked_img = []
    alpha = 0.3
    masks[masks < 0.5]  = 0
    for img_i in range(args.topk):
        ori_img = ori_imgs[img_i].cpu().numpy().transpose(1, 2, 0)
        # normal masked
        mask = masks[img_i].permute(1, 2, 0)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        masked_img.append(mask * ori_img)
    masked_imgs = np.concatenate(masked_img, axis = 1)
    if args.split:
        if args.reverse:
            plt.imsave(fname = os.path.join(f"./{__file__[:-3]}_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.topk}/", f'l{layer_i + 1}_{concept_i + 1}_masked_reverse.png'), arr = masked_imgs, vmin = 0.0, vmax = 1.0)
        else:
            plt.imsave(fname = os.path.join(f"./{__file__[:-3]}_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.topk}/", f'l{layer_i + 1}_{concept_i + 1}_masked.png'), arr = masked_imgs, vmin = 0.0, vmax = 1.0)
    return masked_imgs        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument("--device", default = "0", type = str)
    parser.add_argument('--topk', type = int, choices = [1, 3, 5, 9, 16, 25], required = True)
    parser.add_argument('--split', action = "store_true", default = False, help = "Store each concept individually")
    parser.add_argument('--individually', action = "store_true", default = False)
    parser.add_argument('--eigen_topk', default = 1, type = int)
    parser.add_argument('--heatmap', action = "store_true", default = False)
    parser.add_argument('--masked', action = "store_true", default = False)
    parser.add_argument('--use_CLS_token', action = "store_true", default = False)
    parser.add_argument('--resp_mode', default = "avg", choices=['avg', 'max'], type = str)
    parser.add_argument('--wo_norm', default = False, action = "store_true")
    parser.add_argument("--param_root", type = str, required = True)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.saved_dir = f"./find_topk_area_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.topk}"
    print(args)
    
    nrow = args.topk
    
    layer_sizes, image_size = get_layer_img_size(args.basic_model)
    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                        transforms.CenterCrop((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    data_path = ""
    all_prototype_tensors = []
    all_mask_tensors = []
    all_ori_imgs = []
    with torch.no_grad():
        os.makedirs(f"{args.saved_dir}/", exist_ok = True)
        os.makedirs(f"{args.saved_dir}/ori_img", exist_ok = True)
        os.makedirs(f"{args.saved_dir}/masked", exist_ok = True)

        data_path, train_path, val_path, num_classes = get_dataset(args.case_name)
        data_path = data_path + train_path

        concept_covs = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_covs"]
        concept_means = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_means"]
        concept_vecs, concept_means = load_concept(concept_covs, concept_means, args.eigen_topk)
        
        extra_args = {}
        if "MCPNet_pp" in args.model:
            extra_args["sel_layers"] = args.sel_layers
            model = load_model(model = args.model, basic_model = args.basic_model, num_classes = num_classes, \
                            concept_per_layer = args.concept_per_layer, concept_cha = args.concept_cha, **extra_args)
        else:
            model = load_model(model = args.model, basic_model = args.basic_model, num_classes = num_classes, **extra_args)
      
        trained_param_path = f"{args.param_root}/{args.case_name}/{args.basic_model}/best_model.pkl"
        load_weight(model, trained_param_path)

        model.cuda()
        model.eval()

        max_resp_index = torch.load(f"./find_topk_response_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/max_resp_value_idx.pkl", map_location = f"cuda:0")
        max_resp_value = torch.load(f"./find_topk_response_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/max_resp_value.pkl", map_location = f"cuda:0")

        train_dataset = customDataset(data_path, data_transforms)
        topk_prototypes = []
        topk_prototype_masks = []
        topk_prototype_imgs = []
        target_layer = [2, 5, 8, 11]
        for layer_i, (index, f_size) in tqdm.tqdm(enumerate(zip(max_resp_index, layer_sizes)), total = len(max_resp_index)):
            image_ids = (index // (f_size * f_size)).type(torch.int64) 

            for concept_i in range(index.shape[0]):
                top_i = 0
                shown_img_idxs = []

                imgs = []
                labels = []
                ori_imgs = []
                paths = []
                while len(shown_img_idxs) < args.topk:
                    while image_ids[concept_i, top_i] in shown_img_idxs:
                        top_i += 1
                        if top_i >= image_ids.shape[1]:
                            break
                    shown_img_idxs.append(image_ids[concept_i, top_i])

                    img, label, ori_img, path = train_dataset[image_ids[concept_i, top_i]]
                        
                    imgs.append(img)
                    labels.append(label)
                    ori_imgs.append(ori_img)
                    if args.individually:
                        ori_img = ori_img.permute(1, 2, 0)
                        ori_img = Image.fromarray((ori_img * 255).numpy().astype(np.uint8))
                        ori_img.save(f"{args.saved_dir}/ori_img/l{layer_i + 1}_{concept_i + 1}_{len(ori_imgs)}.png")
                    paths.append(path)

                imgs = torch.stack(imgs, dim = 0)
                labels = torch.tensor(labels)
                ori_imgs = torch.stack(ori_imgs, dim = 0)
                for i in range(len(args.concept_cha)):
                    concept_vecs[i] = concept_vecs[i].type(torch.float32).cuda()
                    concept_means[i] = concept_means[i].type(torch.float32).cuda()
                outputs = model(imgs.cuda())
                feat = outputs[0][layer_i]
                concept_num = args.concept_per_layer[layer_i]
                cha_per_con = args.concept_cha[layer_i]

                if len(feat.shape) == 3:
                    feat = feat[:, 1:].permute(0, 2, 1)
                    B, C, D = feat.shape
                    feat = feat.reshape(B, concept_num, cha_per_con, D)
                else:
                    B, C, H, W = feat.shape
                    feat = feat.reshape(B, concept_num, cha_per_con, H, W)
                    feat = feat.flatten(3)
                    D = H * W

                feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
                resps = torch.sum(feat[:, concept_i : concept_i + 1] * concept_vecs[layer_i][None, concept_i : concept_i + 1, :, None], dim = 2)
                resps = resps.reshape(B, 1, f_size, f_size)
                resps = torch.abs(resps)

                # remove negative weight ===============
                # or minimax ---------------------------
                resps_min = -torch.nn.functional.adaptive_max_pool2d(-resps, output_size = 1)
                resps = resps - resps_min
                resps_max = torch.nn.functional.adaptive_max_pool2d(resps, output_size = 1)
                resps = resps / resps_max
                masks = resps
                masks = torch.nn.functional.interpolate(masks, size = (image_size, image_size), mode = "bicubic", align_corners = False)
                topk_prototype = ori_imgs.cuda()
                
                if args.split:
                    save_image(topk_prototype, f"./{__file__[:-3]}_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.topk}/l{layer_i + 1}_{concept_i + 1}.png", nrow = nrow)
                if args.heatmap:
                    save_heatmap(ori_imgs, masks, args, layer_i = layer_i, concept_i = concept_i)
                if args.masked:
                    save_masked(ori_imgs, masks, args)
                else:
                    topk_prototypes.append(topk_prototype.cpu())
                    topk_prototype_masks.append(masks.cpu())
                    topk_prototype_imgs.append(ori_imgs.cpu())

                if args.individually:
                    topk_prototype_ide = topk_prototype.permute(0, 2, 3, 1)
                    for i in range(topk_prototype_ide.shape[0]):
                        img = topk_prototype_ide[i]
                        img[img < 0] = 0
                        img = (img * 255).cpu().numpy().astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(f"{args.saved_dir}/masked/l{layer_i + 1}_{concept_i + 1}_{i + 1}_masked.png")
                            
            if not args.split:
                vis_prototypes = torch.cat(topk_prototypes, dim = 0)
                all_prototype_tensors.append(vis_prototypes.reshape(-1, args.topk, vis_prototypes.shape[1], vis_prototypes.shape[2], vis_prototypes.shape[3]))
                vis_masks = torch.cat(topk_prototype_masks, dim = 0)
                all_mask_tensors.append(vis_masks.reshape(-1, args.topk, vis_masks.shape[1], vis_masks.shape[2], vis_masks.shape[3]))
                vis_ori_imgs = torch.cat(topk_prototype_imgs, dim = 0)
                all_ori_imgs.append(vis_ori_imgs.reshape(-1, args.topk, vis_ori_imgs.shape[1], vis_ori_imgs.shape[2], vis_ori_imgs.shape[3]))
                save_image(vis_prototypes, f"{args.saved_dir}/all_prototypes_{layer_i}.png", nrow = nrow)
                if args.heatmap:
                    heatmap_imgs = []
                    for prototype_i in range(concept_num):
                        heatmap = save_heatmap(topk_prototype_imgs[prototype_i], vis_masks[prototype_i * args.topk:(prototype_i + 1) * args.topk], args)
                        heatmap_imgs.append(heatmap)
                    heatmap_imgs = np.concatenate(heatmap_imgs, axis = 0)
                    plt.imsave(fname = os.path.join(args.saved_dir, f'all_prototypes_{layer_i}_heatmap.png'), arr = heatmap_imgs, vmin = 0.0, vmax = 1.0)

                if args.masked:
                    masked_imgs = []
                    for prototype_i in range(concept_num):
                        masked = save_masked(topk_prototype_imgs[prototype_i], vis_masks[prototype_i * args.topk:(prototype_i + 1) * args.topk], args)
                        masked_imgs.append(masked)
                    masked_imgs = np.concatenate(masked_imgs, axis = 0)
                    plt.imsave(fname = os.path.join(args.saved_dir, f'all_prototypes_{layer_i}_masked.png'), arr = masked_imgs, vmin = 0.0, vmax = 1.0)
                topk_prototypes.clear()
                topk_prototype_masks.clear()
                topk_prototype_imgs.clear()
                torch.cuda.empty_cache()
                
        if not args.split:
            all_prototype_tensors = torch.cat(all_prototype_tensors, dim = 0).cpu()
            all_mask_tensors = torch.cat(all_mask_tensors, dim = 0).cpu()
            all_ori_imgs = torch.cat(all_ori_imgs, dim = 0).cpu()
            torch.save({"Masked_img" : all_prototype_tensors, "Mask" : all_mask_tensors, "ori_imgs" : all_ori_imgs}, f"{args.saved_dir}/all_prototypes.pkl")