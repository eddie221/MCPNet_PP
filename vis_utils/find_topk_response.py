import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import tqdm
import os
import argparse
from utils import load_model, load_concept, get_dataset, load_weight
from train_utils.arg_reader import basic_args, model_args

if __name__ == "__main__":
    print("Find top-k concept response from the dataset!!!")

    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument("--param_root", type = str, required = True)
    parser.add_argument("--device", default = "0", type = str)
    parser.add_argument('--eigen_topk', default = 1, type = int)
    parser.add_argument('--use_trained', default = False, action = "store_true")
    args = parser.parse_args()
    
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.basic_model == "inceptionv3":
        image_size = 299
    else:
        image_size = 224

    data_transforms = transforms.Compose([transforms.Resize(image_size + 32),
                                     transforms.CenterCrop((image_size, image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    data_path, train_path, val_path, num_classes = get_dataset(args.case_name)
    data_path = data_path + train_path

    train_dataset = ImageFolder(data_path, data_transforms)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = False, num_workers = 8)

    # Load concept vectors and means
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

    # max response
    max_resp_value = [torch.tensor([]).cuda()] * len(args.concept_per_layer)
    max_resp_feat = [torch.tensor([]).cuda()] * len(args.concept_per_layer)
    max_resp_index = [torch.tensor([], dtype = torch.int64).cuda()] * len(args.concept_per_layer)

    # min response
    min_resp_value = [torch.tensor([]).cuda()] * len(args.concept_per_layer)
    min_resp_index = [torch.tensor([], dtype = torch.int64).cuda()] * len(args.concept_per_layer)

    args.save_path = f"./find_topk_response_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}"
    print(args.save_path)
    os.makedirs(args.save_path, exist_ok = True)
    N = [0] * len(args.concept_per_layer)
    with torch.no_grad():
        model.eval()
        tmp = []
        for iteration, (img, label) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            img = img.cuda()
            output = model(img)[0]
            features = output

            for layer_i, feat in enumerate(features[:len(args.concept_per_layer)]):
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
                
                # center the features 
                feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
                feat_idx = torch.arange(B * D).reshape(1, -1).cuda() + N[layer_i]
                feat = torch.flatten(feat.permute(1, 2, 0, 3), 2)
                con_response = torch.sum(feat * concept_vecs[layer_i].unsqueeze(-1), dim = 1)
                
                # store the max response    
                max_resp_value[layer_i] = torch.cat([max_resp_value[layer_i], con_response], dim = 1)
                max_resp_index[layer_i] = torch.cat([max_resp_index[layer_i], feat_idx.repeat(max_resp_value[layer_i].shape[0], 1)], dim = 1)
                max_resp_feat[layer_i] = torch.cat([max_resp_feat[layer_i], feat], dim = -1)
                topkv, topki = torch.topk(max_resp_value[layer_i], k = min(int(len(train_dataset)), max_resp_value[layer_i].shape[1]), dim = 1)
                max_resp_value[layer_i] = topkv
                max_resp_index[layer_i] = torch.gather(max_resp_index[layer_i], dim = 1, index = topki)
                max_resp_feat[layer_i] = torch.gather(max_resp_feat[layer_i], dim = -1, index = topki.unsqueeze(1).repeat(1, args.concept_cha[layer_i], 1))

                # store the min response
                min_resp_value[layer_i] = torch.cat([min_resp_value[layer_i], con_response], dim = 1)
                min_resp_index[layer_i] = torch.cat([min_resp_index[layer_i], feat_idx.repeat(min_resp_value[layer_i].shape[0], 1)], dim = 1)
                topkv, topki = torch.topk(min_resp_value[layer_i], k = min(int(len(train_dataset)), min_resp_value[layer_i].shape[1]), dim = 1, largest = False)
                min_resp_value[layer_i] = topkv
                min_resp_index[layer_i] = torch.gather(min_resp_index[layer_i], dim = 1, index = topki)

                N[layer_i] += feat.shape[2]

    reversed_mask = []
    for layer_i in range(len(max_resp_value)):
        reversed_mask.append(max_resp_value[layer_i][:, 0] > min_resp_value[layer_i][:, 0])
        print(reversed_mask[-1].shape)

    torch.save(max_resp_value, f"{args.save_path}/max_resp_value.pkl")
    torch.save(max_resp_index, f"{args.save_path}/max_resp_value_idx.pkl")
    torch.save(min_resp_value, f"{args.save_path}/min_resp_value.pkl")
    torch.save(min_resp_index, f"{args.save_path}/min_resp_value_idx.pkl")
    torch.save(reversed_mask, f"{args.save_path}/reversed_mask.pkl")