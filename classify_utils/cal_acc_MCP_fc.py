import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import torchvision
import numpy as np
import tqdm
import os
import time
import argparse
from PIL import Image
from utils import get_layer_img_size, get_dataset, load_weight, load_model, load_concept
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

def cal_acc(model, concept_vecs, concept_means, data_transforms, data_path, args):
    print("Calculate FC layer accuracy!")
    dataset = customDataset(data_path, transform = data_transforms)
    print("Number of classes : ", len(dataset.classes))
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers = 8)
    total_count = 0
    total_correct = 0
    total_correct5 = 0
    fc_correct = 0
    correct_path = []
    wrong_path = []
    with torch.no_grad():
        remove_concepts = []
        for remove_layer in args.remove_layers:
            remove_concepts.append(np.arange(args.concept_per_layer[remove_layer]) + np.sum(np.concatenate([[0], args.concept_per_layer], axis = 0)[:remove_layer + 1]))
        if len(remove_concepts) != 0:
            remove_concepts = np.concatenate(remove_concepts, axis = 0)
        keep_concepts = np.arange(np.sum(args.concept_per_layer))
        if len(remove_concepts) != 0:
            keep_concepts = np.delete(keep_concepts, remove_concepts)

        keep_concepts = torch.tensor(keep_concepts).cuda()
    
        for iteration, (img, label, image_path) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
            total_count += img.shape[0]
            img = img.cuda()
            outputs = model(img, concept_vecs, concept_means)
            
            logits = outputs[-2]
            
            top1 = torch.topk(logits, dim = 1, k = 1)[1]
            top5 = torch.topk(logits, dim = 1, k = 5)[1]
            correct_resp = (top1 == label.cuda().unsqueeze(1)).sum()
            correct_resp5 = (top5 == label.cuda().unsqueeze(1)).sum()

            total_correct += correct_resp
            total_correct5 += correct_resp5
            image_path = np.array(image_path)
            correct_path.append(image_path[(top1[:, 0].cpu() == label)])
            wrong_path.append(image_path[(top1[:, 0].cpu() != label)])
    
    fc_correct = fc_correct / total_count
    acc_top1 = total_correct / total_count
    acc_top5 = total_correct5 / total_count
    # print(np.concatenate(correct_path))
    os.makedirs(f"./cal_acc_MCP_fc/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}", exist_ok = True)
    np.savetxt(f"./cal_acc_MCP_fc/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/correct.txt", np.concatenate(correct_path), fmt="%s")
    np.savetxt(f"./cal_acc_MCP_fc/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/wrong.txt", np.concatenate(wrong_path), fmt="%s")
    print(f"Accuracy top1: {acc_top1 * 100:.4f}% ({acc_top1}) top5: {acc_top5 * 100:.4f}% ({acc_top5})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument('--eigen_topk', default = 1, type = int)
    parser.add_argument("--device", default = "0", type = str)
    parser.add_argument("--param_root", type = str, required = True)
    parser.add_argument('--remove_layers', default = [], type = int, nargs = "+", help = "Select the concept to drop. (Start from 0)")
    args = parser.parse_args()

    print("Calculate tree accuracy !!")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    layer_sizes, image_size = get_layer_img_size(args.basic_model)
    print(args)

    data_path, train_path, val_path, num_classes = get_dataset(args.case_name)
    print(data_path)

    # load model
    extra_args = {}
    if "MCPNet_pp" in args.model:
        extra_args["sel_layers"] = args.sel_layers
        model = load_model(model = args.model, basic_model = args.basic_model, num_classes = num_classes, \
                        concept_per_layer = args.concept_per_layer, concept_cha = args.concept_cha, **extra_args)
    else:
        model = load_model(model = args.model, basic_model = args.basic_model, num_classes = num_classes, **extra_args)

    load_weight(model, f"{args.param_root}/{args.case_name}/{args.basic_model.lower()}/best_model.pkl")
    model.eval()

    # Load concept vectors and means
    concept_covs = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_covs"]
    concept_means = torch.load(f"{args.param_root}/{args.case_name}/{args.basic_model}/MCP_data.pkl", weights_only = False)["concept_means"]
    concept_vecs, concept_means = load_concept(concept_covs, concept_means, args.eigen_topk)

    for i in range(len(concept_vecs)):
        concept_vecs[i] = concept_vecs[i].type(torch.float32)
        concept_means[i] = concept_means[i].type(torch.float32)


    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    start = time.time()
    cal_acc(model, concept_vecs, concept_means, data_transforms, data_path + val_path, args)
    print("Times : {} hrs".format((time.time() - start) / 3600))

