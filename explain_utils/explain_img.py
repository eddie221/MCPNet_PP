import torch
import torchvision.transforms as transforms
import tqdm
import numpy as np
import os
import argparse
from PIL import Image
from torchvision.utils import save_image
import time
import cv2
import matplotlib.pyplot as plt
import sys
import math
import pandas as pd

from utils import info_log, load_weight, load_model, load_concept, get_dataset, get_layer_img_size, id2name
from train_utils import basic_args, model_args
import glob

def cal_MCP(feats, concept_vecs, concept_means, args):
    max_responses = []
    response_mask = []
    for layer_i, feat in enumerate(feats):
        concept_num = args.concept_per_layer[layer_i]
        cha_per_con = args.concept_cha[layer_i]
        if len(feat.shape) == 3:
            feat = feat.permute(0, 2, 1)
            feat_d = 3
        else:
            feat = feat.flatten(2)
            feat_d = 4
        B, C, N = feat.shape
        feat = feat.reshape(B, concept_num, cha_per_con, N)
        feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
        # calculate concept vector from covariance matrix
        concept_vector = concept_vecs[layer_i].cuda()
        response = torch.sum(feat * concept_vector.unsqueeze(0).unsqueeze(3), dim = 2)
        if feat_d == 3:
            response = response[..., 1:]
        max_response = torch.nn.functional.adaptive_max_pool1d(response, output_size = 1)[..., 0]
        response = response.reshape(B, concept_num, int(math.sqrt(N)), int(math.sqrt(N)))
        response_mask.append(response)
        max_responses.append(max_response)

    MCP_feat = torch.cat([max_responses[i] for i in range(len(max_responses))], dim = -1)
    # MCP_feat = (MCP_feat + 1) / 2
    MCP_feat_norm = MCP_feat / torch.sum(MCP_feat, dim = -1, keepdim = True)
    return MCP_feat_norm, MCP_feat, response_mask

# =============================================================================
# Interpret the given image (few images)
# =============================================================================
def interpret_img(model, data_transform, args):
    masked_imgs = []
    masks = []
    masks_non_resize = []
    # Load image ------------------------------------------------------------
    img = Image.open(args.image_path)
    img.save(os.path.join(args.saved_dir, 'ori_img.png'))
    img_tensor = data_transform(img).cuda().unsqueeze(0)
    ori_img_tensor = img
    for trans in data_transform.transforms[:-1]:
        ori_img_tensor = trans(ori_img_tensor)
    ori_img_tensor = ori_img_tensor.cuda()
    # -------------------------------------------------------------------------

    # Calculate MCP distribution ----------------------------------------------
    with torch.no_grad():
        for i in range(len(concept_vecs)):
            concept_vecs[i] = concept_vecs[i].type(torch.float32)
            concept_means[i] = concept_means[i].type(torch.float32)
        feats = model(img_tensor, concept_vecs, concept_means)[0]

        MCP_feat_norm, MCP_feat, response_mask = cal_MCP(feats, concept_vecs, concept_means, args)
        for layer_i in range(len(response_mask)):
            mask = response_mask[layer_i]

            # normalize between 0~1
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 2 - 1
            
            mask_resized = torch.nn.functional.interpolate(mask, size = ori_img_tensor.shape[1:], mode = "bicubic", align_corners = True)
            mask_resized[mask_resized < 0] = 0
            mask_resized[mask_resized > 1] = 1

            for concept_i in range(mask.shape[1]):
                masked_img = mask_resized[:, concept_i] * ori_img_tensor
                r = torch.round(mask[:, concept_i].max(), decimals = 4)
                if args.save_each:
                    save_image(masked_img, f"{args.saved_dir}/L{layer_i + 1}_{concept_i + 1}_{r}.png")
                masked_imgs.append(masked_img.cpu())
                masks.append(mask_resized[:, concept_i].cpu())
                masks_non_resize.append(mask[:, concept_i].cpu())
    
        masked_imgs = torch.stack(masked_imgs, dim = 0)
        plt.imsave(fname = os.path.join(args.saved_dir, 'img.png'), arr = ori_img_tensor.permute(1, 2, 0).cpu().numpy(), vmin = 0.0, vmax = 1.0)
    return ori_img_tensor, MCP_feat_norm, MCP_feat, masked_imgs, masks, masks_non_resize

class Saver(object):
    def __init__(self, ori_img, masked_imgs, masks, masks_non_resizes, saved_root, vis_prototypes, args, layer_sizes, concept_captions = None):
        self.ori_img = ori_img.cpu().numpy().transpose(1, 2, 0)
        self.masked_imgs = masked_imgs
        self.masks = masks
        self.masks_non_resizes = masks_non_resizes
        self.saved_root = saved_root
        self.concept_per_layer = args.concept_per_layer
        self.log_file = args.log_file
        self.record = args.record
        self.save_topk_prototypes = args.save_topk_prototypes
        self.vis_prototypes = vis_prototypes
        self.layer_sizes = layer_sizes
        self.concept_captions = concept_captions

    def save_crucial_prototype(self, rel_score, ref_score, prototype_list, eval_name, img_ps = None, rel_score_rate = None, ref_score_rate = None):
        '''
            rel_score : the score of concept calculate by specific method 
            ref_score : the score of concept from class MCP calculate by specific method 
            prototypes_list : the ordered list of the concept prototypes
            sub_folder : the sub folder name to store the result
        '''
        if img_ps is None:
            img_ps = ""
        else:
            img_ps = img_ps + " "
        saved_path = os.path.join(self.saved_root, eval_name)
        os.makedirs(saved_path, exist_ok = True)
        with open(self.log_file, "a") as f:
            for closestP_i, prototype_i in enumerate(prototype_list):
                prototype_name = id2name(prototype_i.clone(), self.concept_per_layer)
                if rel_score_rate is not None:
                    print(f"{f'{closestP_i}-th {img_ps}:' : ^30} {prototype_name:<6} -- Related Score : {rel_score[prototype_i]:.6f} ({rel_score_rate[prototype_i]:.6f}) Reference Score : {ref_score[prototype_i]:.6f} ({ref_score_rate[prototype_i]:.6f})", file = f)
                else:
                    print(f"{f'{closestP_i}-th {img_ps}:' : ^30} {prototype_name:<6} -- Related Score : {rel_score[prototype_i]:<8.6f} Reference Score : {ref_score[prototype_i]:.6f}", file = f)
                    
                if (closestP_i < self.save_topk_prototypes):
                    layer_i = int(prototype_name.split("_")[0][1:]) - 1

                    # with masked on explained image
                    # save_image(torch.cat([self.masked_imgs[prototype_i].unsqueeze(0), self.vis_prototypes[prototype_i]], dim = 0), os.path.join(saved_path, f"{img_ps} {closestP_i}th {prototype_name} {value:.4f}_{ref_score[prototype_i]:.4f}.png"), nrow = 1 + 5)
                    # without masked on explained image
                    result = Image.fromarray(self.vis_prototypes[prototype_i].astype(np.uint8))
                    result.save(os.path.join(saved_path, f"{img_ps} {closestP_i}th {prototype_name} {rel_score[prototype_i]:.4f}_{ref_score[prototype_i]:.4f}.png"))

                    mask = self.masks[prototype_i]
                    mask = torch.clip(mask, max = 1, min = -1)
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask[0]), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap)/255
                    heatmap = heatmap[...,::-1] # OpenCV's BGR to RGB
                    heatmap_img =  0.2 * np.float32(heatmap) + 0.6 * np.float32(self.ori_img)
                    plt.imsave(fname = os.path.join(saved_path, f'heatmap_{prototype_name}.png'), arr = heatmap_img, vmin = 0.0, vmax = 1.0)
                    max_val, max_idx = torch.nn.functional.adaptive_max_pool2d(self.masks_non_resizes[prototype_i].unsqueeze(0), output_size = (1, 1), return_indices = True)
                    max_idx = max_idx[0, 0, 0, 0]
                    y_s = (max_idx // self.layer_sizes[layer_i]) * 224 // self.layer_sizes[layer_i]
                    y_e = (max_idx // self.layer_sizes[layer_i] + 1) * 224 // self.layer_sizes[layer_i]
                    x_s = (max_idx % self.layer_sizes[layer_i]) * 224 // self.layer_sizes[layer_i]
                    x_e = (max_idx % self.layer_sizes[layer_i] + 1) * 224 // self.layer_sizes[layer_i]
                    patch = self.ori_img[y_s : y_e, x_s : x_e, :]
                    plt.imsave(fname = os.path.join(saved_path, f'patch_{prototype_name}.png'), arr = patch, vmin = 0.0, vmax = 1.0)
                    cv_img = (self.ori_img * 255)
                    cv_img = np.ascontiguousarray(cv_img, dtype=np.uint8)
                    if max_val.flatten() != 0:
                        annotated_img = cv2.rectangle(cv_img, (x_s.item(), y_s.item()), (x_e.item(), y_e.item()), (255, 0, 0), 3)
                    else:
                        annotated_img = cv_img
                    plt.imsave(fname = os.path.join(saved_path, f'ret_patch_{prototype_name}.png'), arr = annotated_img, vmin = 0.0, vmax = 1.0)
                    # sys.exit()

    def save_crucial_prototype_per_layer(self, rel_score, ref_score, prototype_list, eval_name, img_ps = None, rel_score_rate = None, ref_score_rate = None):
        '''
            rel_score : the score of concept calculate by specific method 
            ref_score : the score of concept from class MCP calculate by specific method 
            prototypes_list : the ordered list of the concept prototypes
            sub_folder : the sub folder name to store the result
        '''
        if img_ps is None:
            img_ps = ""
        else:
            img_ps = img_ps + " "
        saved_path = os.path.join(self.saved_root, eval_name)
        os.makedirs(saved_path, exist_ok = True)
        shown_layers = []
        color_palette = [[255., 0, 0], [0, 255., 0], [0, 0, 255.], [255., 165., 0]]
        overlap_heat_maps = np.zeros_like(self.ori_img)
        overlap_binary_mask = np.zeros((self.ori_img.shape[0], self.ori_img.shape[1]))
        with open(self.log_file, "a") as f:
            for closestP_i, prototype_i in enumerate(prototype_list):
                prototype_name = id2name(prototype_i.clone(), self.concept_per_layer)
                layer_i = int(prototype_name.split("_")[0][1:]) - 1

                if layer_i in shown_layers:
                    continue
                else:
                    shown_layers.append(layer_i)
                
                if rel_score_rate is not None:
                    print(f"{f'{closestP_i}-th {img_ps}:' : ^30} {prototype_name:<6} -- Image value : {rel_score[prototype_i]:.6f} ({rel_score_rate[prototype_i]:.6f}) Reference value : {ref_score[prototype_i]:.6f} ({ref_score_rate[prototype_i]:.6f})", file = f)
                else:
                    print(f"{f'{closestP_i}-th {img_ps}:' : ^30} {prototype_name:<6} -- Image value : {rel_score[prototype_i]:<8.6f} Reference value : {ref_score[prototype_i]:.6f}", file = f)

                # with masked on explained image
                # save_image(torch.cat([self.masked_imgs[prototype_i].unsqueeze(0), self.vis_prototypes[prototype_i]], dim = 0), os.path.join(saved_path, f"{img_ps} {closestP_i}th {prototype_name} {value:.4f}_{ref_score[prototype_i]:.4f}.png"), nrow = 1 + 5)
                # without masked on explained image
                result = Image.fromarray(self.vis_prototypes[prototype_i].astype(np.uint8))
                result.save(os.path.join(saved_path, f"{img_ps} {closestP_i}th {prototype_name} {rel_score[prototype_i]:.4f}_{ref_score[prototype_i]:.4f}.png"))

                mask = self.masks[prototype_i][0]
                mask = torch.clip(mask, max = 1, min = -1)
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap)/255
                binary_mask = mask
                binary_mask[binary_mask >= 0.5] = 1
                binary_mask[binary_mask < 0.5] = 0
                overlap_binary_mask = overlap_binary_mask + np.array(binary_mask)
                overlap_heat_maps = overlap_heat_maps + np.array(binary_mask[..., None], dtype = np.float32) * np.array(color_palette[layer_i], dtype = np.float32) * 0.5
                heatmap = heatmap[...,::-1] # OpenCV's BGR to RGB
                heatmap_img =  0.3 * np.float32(heatmap) + 0.7 * np.float32(self.ori_img)
                plt.imsave(fname = os.path.join(saved_path, f'heatmap_{prototype_name}.png'), arr = heatmap_img, vmin = 0.0, vmax = 1.0)
                max_val, max_idx = torch.nn.functional.adaptive_max_pool2d(self.masks_non_resizes[prototype_i].unsqueeze(0), output_size = (1, 1), return_indices = True)
                max_idx = max_idx[0, 0, 0, 0]
                y_s = (max_idx // self.layer_sizes[layer_i]) * 224 // self.layer_sizes[layer_i]
                y_e = (max_idx // self.layer_sizes[layer_i] + 1) * 224 // self.layer_sizes[layer_i]
                x_s = (max_idx % self.layer_sizes[layer_i]) * 224 // self.layer_sizes[layer_i]
                x_e = (max_idx % self.layer_sizes[layer_i] + 1) * 224 // self.layer_sizes[layer_i]
                patch = self.ori_img[y_s : y_e, x_s : x_e, :]
                plt.imsave(fname = os.path.join(saved_path, f'patch_{prototype_name}.png'), arr = patch, vmin = 0.0, vmax = 1.0)
                cv_img = (self.ori_img * 255)
                cv_img = np.ascontiguousarray(cv_img, dtype=np.uint8)
                if max_val.flatten() != 0:
                    annotated_img = cv2.rectangle(cv_img, (x_s.item(), y_s.item()), (x_e.item(), y_e.item()), (255, 0, 0), 3)
                else:
                    annotated_img = cv_img
                plt.imsave(fname = os.path.join(saved_path, f'ret_patch_{prototype_name}.png'), arr = annotated_img, vmin = 0.0, vmax = 1.0)
                # sys.exit()

                if len(shown_layers) == 4:
                    break
        base_img = (self.ori_img * 255).astype(np.uint8)
        overlap_heat_maps = np.clip(overlap_heat_maps.astype(np.uint8), 0, 255).astype(np.uint8)
        overlap_result = cv2.addWeighted(base_img, 1.0, overlap_heat_maps, 1.0, 0)
        plt.imsave(fname = os.path.join(saved_path, f'heatmap.png'), arr = overlap_result, vmin = 0, vmax = 255)

    # output the meta data for analysis and simple test
    def save_contained_prototype_eval(self, rel_score, ref_score, fc_weight, prototype_list, eval_name, rel_score_rate, ref_score_rate):
        saved_path = os.path.join(self.saved_root, eval_name)
        os.makedirs(saved_path, exist_ok = True)
        
        concept_meta_data = {"Name" : [],
                             "MCP_feat" : [],
                             "gt_MCP_feat" : [],
                             "fc_weight" : [],
                             "MCP std rate" : [],
                             "gt MCP std rate" : []}
        for _, prototype_i in enumerate(prototype_list):
            prototype_name = id2name(prototype_i.clone(), self.concept_per_layer)
            concept_meta_data["Name"].append(prototype_name)
            concept_meta_data["MCP_feat"].append(rel_score[prototype_i].item())
            concept_meta_data["gt_MCP_feat"].append(ref_score[prototype_i].item())
            concept_meta_data["fc_weight"].append(fc_weight[prototype_i].item())
            concept_meta_data["MCP std rate"].append(rel_score_rate[prototype_i].item())
            concept_meta_data["gt MCP std rate"].append(ref_score_rate[prototype_i].item())

        df = pd.DataFrame(concept_meta_data)
        df.to_csv(os.path.join(saved_path, f'meta_data.csv'))

    def save_contained_prototype_filtered(self, rel_score, fc_weight, prototype_list, eval_name, rel_score_rate, ref_score_rate):
        saved_path = os.path.join(self.saved_root, eval_name)
        os.makedirs(saved_path, exist_ok = True)
        with open(self.log_file, "a") as f:
            for idx, prototype_i in enumerate(prototype_list):
                # the filter condition
                if fc_weight[prototype_i] < 0.005 or \
                    rel_score_rate[prototype_i] < 0.3:
                    continue
                prototype_name = id2name(prototype_i.clone(), self.concept_per_layer)
                layer_i = int(prototype_name.split("_")[0][1:]) - 1
                print(f"    {f'{idx}-th : {prototype_name}' : ^10} -- Image MCP feat: {rel_score[prototype_i]:10.6f} Image MCP feat (std.): {rel_score_rate[prototype_i]:10.6f} Class MCP feat (std.): {ref_score_rate[prototype_i]:10.6f} FC weight: {fc_weight[prototype_i]:10.6f}", file = f)

                if self.concept_captions is not None:
                    concepts = self.concept_captions.loc[prototype_i.item()]
                    C1 = concepts["Concept_1"]
                    C2 = concepts["Concept_2"]
                    C3 = concepts["Concept_3"]
                    print(f"            C1:{C1} C2:{C2} C3:{C3}", file = f)

                # with masked on explained image
                # save_image(torch.cat([self.masked_imgs[prototype_i].unsqueeze(0), self.vis_prototypes[prototype_i]], dim = 0), os.path.join(saved_path, f"{eval_name} {idx}th {prototype_name} {rel_score[prototype_i]:.4f}_{ref_score[idx]:.4f}.png"), nrow = 1 + 5)
                # without masked on explained image
                result = Image.fromarray(self.vis_prototypes[prototype_i].astype(np.uint8))
                result.save(os.path.join(saved_path,  f"{eval_name} {idx}th {prototype_name} {rel_score[prototype_i]:.4f}_{fc_weight[prototype_i]:.4f}.png"))

                mask = self.masks[prototype_i]
                mask = torch.clip(mask, max = 1, min = -1)
                heatmap = cv2.applyColorMap(np.uint8(255 * mask[0]), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap)/255
                heatmap = heatmap[...,::-1] # OpenCV's BGR to RGB
                heatmap_img =  0.2 * np.float32(heatmap) + 0.6 * np.float32(self.ori_img)
                plt.imsave(fname = os.path.join(saved_path, f'heatmap_{prototype_name}.png'), arr = heatmap_img, vmin = 0.0, vmax = 1.0)

                max_val, max_idx = torch.nn.functional.adaptive_max_pool2d(self.masks_non_resizes[prototype_i].unsqueeze(0), output_size = (1, 1), return_indices = True)
                max_idx = max_idx[0, 0, 0, 0]
                y_s = (max_idx // self.layer_sizes[layer_i]) * 224 // self.layer_sizes[layer_i]
                y_e = (max_idx // self.layer_sizes[layer_i] + 1) * 224 // self.layer_sizes[layer_i]
                x_s = (max_idx % self.layer_sizes[layer_i]) * 224 // self.layer_sizes[layer_i]
                x_e = (max_idx % self.layer_sizes[layer_i] + 1) * 224 // self.layer_sizes[layer_i]
                patch = self.ori_img[y_s : y_e, x_s : x_e, :]
                plt.imsave(fname = os.path.join(saved_path, f'patch_{prototype_name}.png'), arr = patch, vmin = 0.0, vmax = 1.0)
                cv_img = (self.ori_img * 255)
                cv_img = np.ascontiguousarray(cv_img, dtype=np.uint8)
                if max_val.flatten() != 0:
                    annotated_img = cv2.rectangle(cv_img, (x_s.item(), y_s.item()), (x_e.item(), y_e.item()), (255, 0, 0), 3)
                else:
                    annotated_img = cv_img
                plt.imsave(fname = os.path.join(saved_path, f'ret_patch_{prototype_name}.png'), arr = annotated_img, vmin = 0.0, vmax = 1.0)
                print(os.path.join(saved_path, f'ret_patch_{prototype_name}.png'))
                # =====================================================

def large_contributions(fc_weight, MCP_feat, class_MCP, MCP_mean, MCP_std, sel_class, args, eval_name, img_saver):
    img_concept_contribution = (fc_weight * MCP_feat)[sel_class]
    class_concept_contribution = (fc_weight * class_MCP)[sel_class]
    sort_contri_des = torch.sort(img_concept_contribution, descending = True)
    img_con_resp_norm = (MCP_feat - MCP_mean) / MCP_std
    class_con_resp_norm = (class_MCP - MCP_mean) / MCP_std
    
    info_log(f"{eval_name} Large contribution prototypes : ", type = args.record, file = args.log_file)
    info_log(f"Image value (x) => Concept Contribution from Image (Normalized Concept Response from image) \nReference value (x) => Concept Contribution from {sel_class} Class (Normalized Concept Response from {sel_class} class)", type = args.record, file = args.log_file)
    img_saver.save_crucial_prototype(rel_score = img_concept_contribution, 
                                     ref_score = class_concept_contribution, 
                                     prototype_list = sort_contri_des[1], 
                                     eval_name = eval_name, 
                                     img_ps = "large contribution", 
                                     rel_score_rate = img_con_resp_norm[0], 
                                     ref_score_rate = class_con_resp_norm[sel_class])
    info_log(f"==============================\n", type = args.record, file = args.log_file)

def large_contributions_per_layer(fc_weight, MCP_feat, class_MCP, MCP_mean, MCP_std, sel_class, args, eval_name, img_saver):
    img_concept_contribution = (fc_weight * MCP_feat)[sel_class]
    class_concept_contribution = (fc_weight * class_MCP)[sel_class]
    sort_contri_des = torch.sort(img_concept_contribution, descending = True)
    img_con_resp_norm = (MCP_feat - MCP_mean) / MCP_std
    class_con_resp_norm = (class_MCP - MCP_mean) / MCP_std
    
    info_log(f"{eval_name} Large contribution prototypes (per layer) : ", type = args.record, file = args.log_file)
    info_log(f"Image value (x) => Concept Contribution from Image (Normalized Concept Response from image) \nReference value (x) => Concept Contribution from {sel_class} Class (Normalized Concept Response from {sel_class} class)", type = args.record, file = args.log_file)
    img_saver.save_crucial_prototype_per_layer(rel_score = img_concept_contribution, ref_score = class_concept_contribution, prototype_list = sort_contri_des[1], eval_name = eval_name, img_ps = "large contribution l", rel_score_rate = img_con_resp_norm[0], ref_score_rate = class_con_resp_norm[sel_class])
    info_log(f"==============================\n", type = args.record, file = args.log_file)


def classify2correct_Contribution(MCP_mean, MCP_std, class_MCP, MCP_feat, fc_weight, pred_class, args, eval_name):
    class_MCP_weighted = class_MCP * fc_weight
    MCP_feat_weighted = MCP_feat[0] * fc_weight[args.gt_class]

    # calcualte the normalize response weight 
    img_rate = (MCP_feat[0] - MCP_mean) / MCP_std
    gt_class_rate = (class_MCP[args.gt_class] - MCP_mean) / MCP_std
    pred_class_rate = (class_MCP[pred_class] - MCP_mean) / MCP_std
    # (x - mean) / std different
    large_diff = MCP_feat_weighted - class_MCP_weighted[args.gt_class]

    need_larger = torch.sort(large_diff, descending = False)
    info_log(f"Prototypes need increase (contribution, response) : ", type = args.record, file = args.log_file)
    info_log(f"Image value (x) => Concept Contribution from Image (Normalized Concept Response from image) \nReference value (x) => Concept Contribution from GT Class (Normalized Concept Response from GT class)", type = args.record, file = args.log_file)
    img_saver.save_crucial_prototype(rel_score = MCP_feat_weighted, 
                                     ref_score = class_MCP_weighted[args.gt_class], 
                                     prototype_list = need_larger[1], 
                                     eval_name = eval_name, 
                                     img_ps = "need increase", 
                                     rel_score_rate = img_rate, 
                                     ref_score_rate = gt_class_rate)
    info_log(f"==============================\n", type = args.record, file = args.log_file)

    info_log(f"Prototypes need increase (contribution, response, check response to pred) : ", type = args.record, file = args.log_file)
    info_log(f"Image value (x) => Concept Contribution from Image (Normalized Concept Response from image) \nReference value (x) => Concept Contribution from Predicted Class (Normalized Concept Response from Predicted class)", type = args.record, file = args.log_file)
    img_saver.save_crucial_prototype(rel_score = MCP_feat_weighted, 
                                     ref_score = class_MCP_weighted[pred_class], 
                                     prototype_list = need_larger[1], 
                                     eval_name = eval_name, 
                                     img_ps = "need increase (p)", 
                                     rel_score_rate = img_rate, 
                                     ref_score_rate = pred_class_rate)
    info_log(f"==============================\n", type = args.record, file = args.log_file)


def get_args():
    parser = argparse.ArgumentParser(description='MCP interpret', parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument("--device", default = "0", type = str)

    parser.add_argument('--caption_model', default = "gpt-4o", type = str)

    parser.add_argument("--image_paths", default = [], type = str, nargs = "+", required = True, help = "The path of images used to generate the explanation.")
    parser.add_argument("--gt_class", type = int, default = None)

    parser.add_argument('--eigen_topk', default = 1, type = int)

    parser.add_argument("--param_root", type = str, required = True)
    parser.add_argument("--save_each", action = "store_true", default = False, help = "Save each prototypes")
    parser.add_argument("--saved_dir", type = str, help = "Saved directory.", required = True)
    parser.add_argument("--save_topk_prototypes", type = int, default = 3, help = "Save save the topk crucial prototypes")

    # interpret mode
    parser.add_argument("--show_contained_prototype", action = "store_true", default = False, help = "Show the prototypes that response above the given threshold.")
    parser.add_argument("--MCP_threshold", type = float, default = 0.5)

    parser.add_argument("--add_save_prototypes", default = [], type = str, nargs = "+", help = "Manual set the additional saved prototypes")
    parser.add_argument("--record", default = ["log"], type = str, nargs = "+", help = "Output settings.", choices = ["std", "log"])
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = time.time()
    args = get_args()
    print(args)
    args.dst = f"/eva_data_4/bor/MCPNet_plus_dev/pkl/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    save_prototype = []
    for add_save_prototype in args.add_save_prototypes:
        if "_" not in add_save_prototype:
            layer = int(add_save_prototype[1:])
            for i in range(1, args.concept_per_layer[layer] + 1):
                save_prototype.append(f"l{layer}_{i}")
    if len(save_prototype) != 0:
        args.add_save_prototypes = save_prototype

    class_num = get_dataset(args.case_name)[-1]

    # load concept caption -----------------------------------------------------
    concept_captions = pd.read_csv(f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model}/{args.caption_model}/concept caption_v1.csv", index_col = None)

    # Load model (load pretrain if needed) ------------------------------------
    extra_args = {}
    if "MCPNet_pp" in args.model:
        extra_args["sel_layers"] = args.sel_layers
        model = load_model(model = args.model, basic_model = args.basic_model, num_classes = class_num, \
                        concept_per_layer = args.concept_per_layer, concept_cha = args.concept_cha, **extra_args)
    else:
        model = load_model(model = args.model, basic_model = args.basic_model, num_classes = class_num, **extra_args)
    trained_param_path = f"{args.param_root}/{args.case_name}/{args.basic_model}/best_model.pkl"
    load_weight(model, path = trained_param_path)
    model.eval()

    layer_sizes, args.image_size = get_layer_img_size(args.basic_model)
    # -------------------------------------------------------------------------

    # Prepare the transformation ------------------------------------
    data_transform = transforms.Compose([transforms.Resize(args.image_size + 32),
                                     transforms.CenterCrop((args.image_size, args.image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # calculate covariance matrix ---------------------------------------------
    if os.path.isfile(args.image_paths[0]):
        args.image_paths = args.image_paths
    else:
        image_paths = []
        for folder_path in args.image_paths:
            image_paths.extend(sorted(glob.glob(os.path.join(folder_path, "*"))))

        args.image_paths = image_paths
    ori_save_dir = args.saved_dir
    
    prototype_contain_counts = {}
    for image_path in args.image_paths:
        args.image_path = image_path
        image_name = args.image_path.split("/")[-1].split('.')[0]
        args.saved_dir = f"{ori_save_dir}/interpret_result/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{image_name}"
        os.makedirs(args.saved_dir, exist_ok = True)
        args.log_file = f"{args.saved_dir}/log.txt"
        with open(args.log_file, "w") as f:
            print(f"Args : {args}", file = f)

        # generate the prototypes response -----------------------------------------
        ## Load Concept prototypes -------------------------------------------------
        concept_covs = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_covs"]
        concept_means = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_means"]
        concept_vecs, concept_means = load_concept(concept_covs, concept_means, args.eigen_topk)

        ori_img_tensor, img_MCP_norm, img_MCP, masked_imgs, masks, masks_non_resize = interpret_img(model, data_transform, args)
        torch.save(img_MCP, os.path.join(args.saved_dir, "img_MCP.pkl"))
        # --------------------------------------------------------------------------

        # load prototype visualization ---------------------------------------------
        ## the patch images
        vis_prototypes = np.load(f"./extract_patch_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/all_patches.npy", allow_pickle = True)
        print("vis_prototypes : ", vis_prototypes.shape)
        # --------------------------------------------------------------------------

        # load the class MCP ----------------------------------------------
        MCP_features_raw = torch.load(f"./MCP_feats_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/MCP_features_raw.pkl")
        all_labels = torch.load(f"./MCP_feats_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/labels.pkl")
        labels, count = torch.unique(all_labels, return_counts = True)
        print(MCP_features_raw.shape)

        # calculate MCP feautres mean and std.
        MCP_mean = torch.mean(MCP_features_raw, dim = 0)
        MCP_std = torch.std(MCP_features_raw, dim = 0)

        # calculate class MCP features
        class_MCP = []
        for label_i in labels:
            class_MCP.append(torch.mean(MCP_features_raw[all_labels == label_i], dim = 0))
        class_MCP = torch.stack(class_MCP, dim = 0)
        # --------------------------------------------------------------------------

        # classify image -----------------------------------------------------------
        if "vit" in args.basic_model:
            fc_weight = model.fc_patch_MCP.weight.detach()
        else:
            fc_weight = model.fc.weight.detach()
        logits = torch.sum(fc_weight * img_MCP, dim = 1)
        # print(logits)
        pred_class = torch.argmax(logits)
        info_log(f"Classify result : {pred_class}", type = args.record, file = args.log_file)

        print(f"Classify result : {pred_class}")
        # --------------------------------------------------------------------------

        # Draw the MCP features ----------------------------------------------------
        min_value = min(class_MCP[args.gt_class].min().cpu(), class_MCP[pred_class].min().cpu(), img_MCP.min().cpu())
        max_value = max(class_MCP[args.gt_class].max().cpu(), class_MCP[pred_class].max().cpu(), img_MCP.max().cpu())
        fig, ax = plt.subplots()
        ax.set_title("MCP features")
        rects = ax.bar(np.arange(class_MCP.shape[1]) - 0.25, class_MCP[args.gt_class].cpu(), label = "GT MCP", width = 0.25)
        rects = ax.bar(np.arange(class_MCP.shape[1]), class_MCP[pred_class].cpu(), label = "Pred MCP", width = 0.25)
        rects = ax.bar(np.arange(class_MCP.shape[1]) + 0.25, img_MCP[0].cpu(), label = "Img MCP", width = 0.25)
        ax.legend(loc='upper left')
        plt.vlines(x = np.cumsum(args.concept_per_layer)[:3] + 0.3 - 1, ymin = min_value, ymax = max_value, colors = 'red')
        plt.savefig(os.path.join(args.saved_dir, "MCP.png"))
        # --------------------------------------------------------------------------

        # Draw the concept contributions ----------------------------------------------------
        gt_con_cont = class_MCP[args.gt_class] * fc_weight[args.gt_class]
        pred_con_cont = class_MCP[pred_class] * fc_weight[pred_class]
        img_gt_con_cont = img_MCP[0] * fc_weight[args.gt_class]
        img_pred_con_cont = img_MCP[0] * fc_weight[pred_class]
        min_value = min(gt_con_cont.min().cpu(), pred_con_cont.min().cpu(), img_gt_con_cont.min().cpu(), img_pred_con_cont.min().cpu())
        max_value = max(gt_con_cont.max().cpu(), pred_con_cont.max().cpu(), img_gt_con_cont.max().cpu(), img_pred_con_cont.max().cpu())
        fig, ax = plt.subplots()
        ax.set_title("MCP contributions")
        rects = ax.bar(np.arange(gt_con_cont.shape[0]) - 0.2, gt_con_cont.cpu(), label = "GT", width = 0.2)
        rects = ax.bar(np.arange(gt_con_cont.shape[0]), pred_con_cont.cpu(), label = "Pred", width = 0.2)
        rects = ax.bar(np.arange(gt_con_cont.shape[0]) + 0.2, img_gt_con_cont.cpu(), label = "Img (GT)", width = 0.2)
        rects = ax.bar(np.arange(gt_con_cont.shape[0]) + 0.4, img_pred_con_cont.cpu(), label = "Img (Pred)", width = 0.2)
        ax.legend(loc='upper left')
        plt.vlines(x = np.cumsum(args.concept_per_layer)[:3] + 0.6 - 1, ymin = min_value, ymax = max_value, colors = 'red')
        plt.savefig(os.path.join(args.saved_dir, "MCP_cont.png"))
        # --------------------------------------------------------------------------

        # initial saver -------------------------------------------------------------
        img_saver = Saver(ori_img = ori_img_tensor, 
                          masked_imgs = masked_imgs, 
                          masks = masks, 
                          masks_non_resizes = masks_non_resize, 
                          saved_root = args.saved_dir, 
                          vis_prototypes = vis_prototypes, 
                          args = args, 
                          layer_sizes = layer_sizes, 
                          concept_captions = concept_captions)
        # --------------------------------------------------------------------------


        # show contained prototypes ------------------------------------------------
        contained_prototypes_idx = img_MCP >= img_MCP.min()
        print(contained_prototypes_idx)
        print(contained_prototypes_idx.shape)
        contained_prototypes_value = img_MCP[contained_prototypes_idx]
        img_resp_rate = (img_MCP[0] - MCP_mean) / MCP_std
        class_resp_rate = (class_MCP - MCP_mean) / MCP_std
        info_log(f"Image value (x) => Concept Response from Image (Normalized Concept Response from image) \nReference value (x) => Concept Response from GT Class (Normalized Concept Response from GT class)", type = args.record, file = args.log_file)

        img_saver.save_contained_prototype_eval(img_MCP[0], 
                                           class_MCP[args.gt_class],
                                           fc_weight[args.gt_class],
                                           torch.arange(img_MCP.shape[1]).cuda()[contained_prototypes_idx[0]],
                                           eval_name = "contained prototypes (eval)",
                                           rel_score_rate = img_resp_rate,
                                           ref_score_rate = class_resp_rate[args.gt_class]
                                           )

        img_saver.save_contained_prototype_filtered(img_MCP[0], 
                                           fc_weight[args.gt_class],
                                           torch.arange(img_MCP.shape[1]).cuda()[contained_prototypes_idx[0]],
                                           eval_name = "contained prototypes (filtered)",
                                           rel_score_rate = img_resp_rate,
                                           ref_score_rate = class_resp_rate[args.gt_class]
                                           )
        # --------------------------------------------------------------------------
        
        # find large weight * response prototypes -------------------------------------
        print("Ground truth concept weight : \n", fc_weight[args.gt_class])
        print("Ground truth MCP feat : ", class_MCP[args.gt_class])
        print("Image MCP feat : ", img_MCP[0])

        info_log("Ground truth concept weight : \n{}".format(fc_weight[args.gt_class]), type = args.record, file = args.log_file)
        info_log("Ground truth MCP feat : {}".format(class_MCP[args.gt_class]), type = args.record, file = args.log_file)
        info_log("Image MCP feat : {}".format(img_MCP[0]), type = args.record, file = args.log_file)
        if args.gt_class != pred_class:
            large_contributions(fc_weight = fc_weight, 
                                MCP_feat = img_MCP, 
                                class_MCP = class_MCP, 
                                MCP_mean = MCP_mean, 
                                MCP_std = MCP_std, 
                                sel_class = pred_class, 
                                args = args, 
                                eval_name = f"contribution pred {pred_class}", 
                                img_saver = img_saver)
        large_contributions(fc_weight = fc_weight, 
                                MCP_feat = img_MCP, 
                                class_MCP = class_MCP, 
                                MCP_mean = MCP_mean, 
                                MCP_std = MCP_std, 
                                sel_class = args.gt_class, 
                                args = args, 
                                eval_name = f"contribution gt {args.gt_class}", 
                                img_saver = img_saver)
        # ------------------------------------------------------------------------------
        
        # Correct the classification ---------------------------------------------------
        if args.gt_class != pred_class:
            info_log(f"From class {pred_class} to {args.gt_class}", type = args.record, file = args.log_file)
            classify2correct_Contribution(MCP_mean = MCP_mean, 
                                          MCP_std = MCP_std, 
                                          class_MCP = class_MCP, 
                                          MCP_feat = img_MCP, 
                                          fc_weight = fc_weight, 
                                          pred_class = pred_class, 
                                          args = args, 
                                          eval_name = f"Correction {pred_class}_to_{args.gt_class} (Resp)")
        # ------------------------------------------------------------------------------