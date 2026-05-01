from typing import Tuple


def get_dataset(case_name: str) -> Tuple[str, str, str, int]:
    if "AWA2" in case_name:
        print("Using AWA2")
        data_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/"
        train_path, val_path, num_class = "train", "val", 50
    elif "CUB" in case_name:
        print("Using CUB")
        data_path = "/eva_data_4/bor/datasets/CUB_200_2011/"
        train_path, val_path, num_class = "train", "val", 200
    elif "Stanford" in case_name:
        print("Using StanfordCar")
        data_path = "/eva_data_4/bor/datasets/stanford car/"
        train_path, val_path, num_class = "cars_train", "cars_test", 196
    elif "Caltech101" in case_name:
        print("Using Caltech101")
        data_path = "/eva_data_4/bor/datasets/101_ObjectCategories/"
        train_path, val_path, num_class = "train", "val", 101
    elif "Food" in case_name:
        print("Using Food101")
        data_path = "/eva_data_4/bor/datasets/food-101/"
        train_path, val_path, num_class = "train", "test", 101
    elif "SYN" in case_name:
        data_path = "/eva_data_4/bor/datasets/Synthetic/"
        train_path, val_path, num_class = "train/raw", "val/raw", 15
    elif "ImageNet_sub" in case_name:
        data_path = "/eva_data_4/bor/datasets/ImageNet2012/"
        train_path, val_path, num_class = "train_sub", "val_sub", 100
    elif "ImageNet" in case_name:
        data_path = "/eva_data_4/bor/datasets/ImageNet2012/"
        train_path, val_path, num_class = "train", "val", 1000
    elif "Color_shape" in case_name:
        data_path = "/eva_data_4/bor/datasets/MultiColor-Shapes-Database-main/shapes_colors/"
        train_path, val_path, num_class = "train", "val", 9
    elif "aPascal" in case_name:
        data_path = "/eva_data_4/bor/datasets/aPY/aPascal/"
        train_path, val_path, num_class = "train", "test", 20
    elif "LAD_A" in case_name:
        data_path = "/eva_data_4/bor/datasets/LAD/animal/"
        train_path, val_path, num_class = "train", "val", 50
    return data_path, train_path, val_path, num_class


def get_layer_img_size(basic_model):
    image_size = 224
    if basic_model in ("resnet50", "resnet34", "resnet50_relu", "resnet34_relu"):
        layer_sizes = [56, 28, 14, 7]
    elif basic_model == "inceptionv3":
        layer_sizes = [71, 17, 8, 8]
        image_size = 299
    elif basic_model == "mobilenet":
        layer_sizes = [28, 14, 14, 7]
    elif "vit" in basic_model:
        layer_sizes = [14] * 12
    return layer_sizes, image_size
