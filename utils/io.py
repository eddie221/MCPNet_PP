import torch
import importlib


def info_log(message, rank=-1, type=["std"], file=None):
    if rank in [-1, 0]:
        if "std" in type:
            print(message)
        if "log" in type:
            with open(file, "a") as f:
                print(message, file=f)


def load_weight(model: torch.nn.Module, path: str) -> None:
    print(f"load weight from {path}")
    parameters = torch.load(path, map_location='cuda:0')["Model"]
    model = torch.nn.DataParallel(model, device_ids=[0])
    print(model.load_state_dict(parameters, strict=True))


def load_model(model, basic_model, num_classes, **kwargs) -> torch.nn.Module:
    if "resnet" in model.lower():
        model_class = importlib.import_module("models.ResNet")
    elif "inception" in model.lower():
        model_class = importlib.import_module("models.inception_net")
    elif "mobilenet" in model.lower():
        model_class = importlib.import_module("models.mobilenet")
    elif "convnext" in model.lower():
        model_class = importlib.import_module("models.convnext")
    elif "vit" in model.lower():
        model_class = importlib.import_module("models.vit")
    else:
        model_class = importlib.import_module(f"models.{model}")

    model = model_class.load_model(num_classes, basic_model, **kwargs)
    return model
