import argparse
from easydict import EasyDict as edict
import json

def basic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--case_name", type = str, default = None, required = True, help = "Name of the experiments")
    return parser

def model_args():
    parser = argparse.ArgumentParser(add_help=False)
    # MCPNet setting
    parser.add_argument('--concept_per_layer', default = [8, 16, 32, 64], type = int, nargs = "+")
    parser.add_argument('--concept_cha', default = [32, 32, 32, 32], type = int, nargs = "+")
    parser.add_argument("--basic_model", type = str, default = None, required = True, help = "Class name of the model.")
    parser.add_argument('--sel_layers', default = [2, 5, 8, 11], type = int, nargs = "+")
    parser.add_argument("--num_samples_per_class", default = 5, type = int)
    parser.add_argument("--rand_sub", default = False, action = "store_true")
    parser.add_argument('--dropout_p', default = 0.5, type = float)
    return parser

def train_args():
    parser = argparse.ArgumentParser(add_help=False)
    # training hyper parameters
    parser.add_argument("--local-rank", type = int, default = -1, help = "DDP parameter. (Don't modify !!)")
    parser.add_argument("--devices", type = int, default = None, required = True, nargs = "+")
    parser.add_argument("--epoch", type = int, default = 50)
    parser.add_argument("--optimizer", type = str, default = None, required = True, choices = ["adam", "sgd", "adamw"])
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    parser.add_argument("--lr_rate", type = float, default = 100., help = "Learning rate")
    parser.add_argument("--weight_decay", type = float, default = 1e-4)
    parser.add_argument("--lr_scheduler", type = int, default = 20)
    parser.add_argument("--pretrained_path", type = str, default = None, help = "Pretrained parameter path")
    return parser

def dataset_args():
    parser = argparse.ArgumentParser(add_help=False)
    # dataset
    parser.add_argument("--dataloader", type = str, default = "load_data_train_val_classify")
    parser.add_argument("--dataset_name", type = str, default = None, required = True)
    parser.add_argument("--mean", type = float, default = [0.485, 0.456, 0.406])
    parser.add_argument("--std", type = float, default = [0.229, 0.224, 0.225])
    ## training set
    parser.add_argument("--train_batch_size", type = int, default = 64)
    parser.add_argument("--train_num_workers", type = int, default = 8)
    ## validation set
    parser.add_argument("--val_batch_size", type = int, default = 64)
    parser.add_argument("--val_num_workers", type = int, default = 8)
    return parser
    
def read_args():

    cfg = edict()
    parser = argparse.ArgumentParser(description='The options of the MCPNet++.',
                                     parents = [basic_args(), model_args(), train_args(), dataset_args()])

    parser.add_argument("--saved_dir", default = ".", type = str)
    parser.add_argument("--log_type", default = ["std", "log"], type = str, nargs = "+")

    args = vars(parser.parse_args())
    cfg.update(args)
    cfg = edict(cfg)

    if "resnet50" in cfg.basic_model:
        cfg.pretrained_path = "../pretrained/resnet50.pth"
    elif "resnet18" in cfg.basic_model:
        cfg.pretrained_path = "../pretrained/resnet18.pth"
    elif "resnet34" in cfg.basic_model:
        cfg.pretrained_path = "../pretrained/resnet34.pth"
    elif "resnet152" in cfg.basic_model:
        cfg.pretrained_path = "../pretrained/resnet152.pth"
    elif cfg.basic_model == "convnext_base":
        cfg.pretrained_path = "../pretrained/convnext_base_1k_224_ema.pth"
    elif cfg.basic_model == "convnext_small":
        cfg.pretrained_path = "../pretrained/convnext_small_1k_224_ema.pth"
    elif cfg.basic_model == "convnext_tiny":
        cfg.pretrained_path = "../pretrained/convnext_tiny.pth"
    elif cfg.basic_model == "convnext_large":
        cfg.pretrained_path = "../pretrained/convnext_large_1k_224_ema.pth"
    elif cfg.basic_model == "inceptionv3":
        cfg.pretrained_path = "../pretrained/inception_v3.pth"
    elif cfg.basic_model == "densenet121":
        cfg.pretrained_path = "../pretrained/densenet121.pth"
    elif cfg.basic_model == "densenet161":
        cfg.pretrained_path = "../pretrained/densenet161.pth"
    elif "vit" in cfg.basic_model:
        if "vit_t" in cfg.basic_model:
            cfg.pretrained_path = "../pretrained/deit_tiny_16.pth"
        elif "vit_s" in cfg.basic_model:
            cfg.pretrained_path = "../pretrained/deit_small_16.pth"
        elif "vit_b" in cfg.basic_model:
            cfg.pretrained_path = "../pretrained/deit_base_16.pth"
    else:
        assert False, "No pretrained weight !!"


    #############################################################################
    #############################################################################
    # Dataset ###################################################################
    DATASET_ROOT = "/eva_data_4/bor/datasets"
    if cfg.dataset_name == "CUB_200_2011":
        cfg.category = 200
        cfg.train_dataset_path = f"{DATASET_ROOT}/CUB_200_2011/train"
        cfg.val_dataset_path = f"{DATASET_ROOT}/CUB_200_2011/val"
    elif cfg.dataset_name == "AWA2":
        cfg.category = 50
        cfg.train_dataset_path = f"{DATASET_ROOT}/Animals_with_Attributes2/JPEGImages/train"
        cfg.val_dataset_path = f"{DATASET_ROOT}/Animals_with_Attributes2/JPEGImages/val"
    elif cfg.dataset_name == "Caltech101":
        cfg.category = 101
        cfg.train_dataset_path = f"{DATASET_ROOT}/101_ObjectCategories/train"
        cfg.val_dataset_path = f"{DATASET_ROOT}/101_ObjectCategories/val"
    
    if cfg.basic_model != "inceptionv3":
        cfg.train_random_sized_crop = 224
        cfg.val_image_size = 224
    else:
        cfg.train_random_sized_crop = 299
        cfg.val_image_size = 299
    return cfg

def save_args(args):
    with open(f"{args.dst}/args.txt", "w") as f:
        json.dump(args.__dict__, f, indent = 2)
