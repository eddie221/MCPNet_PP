import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from utils.io import load_model as _build_model


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def load_model(args, **extra_kwargs):
    """Instantiate MCPNet_pp and configure it for training (SyncBN, DDP, device)."""
    use_cuda = args.device_id != "cpu"

    model = _build_model(
        model="MCPNet_pp",
        basic_model=args.basic_model,
        num_classes=args.category,
        concept_per_layer=args.concept_per_layer,
        concept_cha=args.concept_cha,
        **extra_kwargs,
    )

    if args.global_rank in [-1, 0]:
        print("Using ", args.device_id)

    if use_cuda and args.global_rank != -1 and has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if "pretrained_path" in args and args.pretrained_path is not None:
        if args.global_rank in [-1, 0]:
            print("load pretrained : {}".format(args.pretrained_path))
        model.load_pretrained(args.pretrained_path)

    model.to(args.device_id)

    if use_cuda and args.global_rank != -1:
        if args.global_rank in [0]:
            print("DDP mode")
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    return model
