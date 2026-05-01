import importlib
import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist
import sys
import time
import shutil

from utils import info_log, cal_cov_component, cal_concept, cal_cov

from train_utils import read_args, save_args, load_model, \
                        CKA_loss_sampled, CCC_with_drop_p, \
                        dataloader, check_device

import torch.nn as nn

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

# =============================================================================
# Run one iteration
# =============================================================================
def train_one_step(model, data, label, loss_funcs, optimizer, args, concept_vectors = None, concept_means = None):
    if isinstance(data, list):
        data = torch.cat(data, dim = 0)
        
    if args.device_id != -1:
        b_data = data.cuda(args.device_id, non_blocking = True)
        b_label = label.cuda(args.device_id, non_blocking = True)
    else:
        b_data = data
        b_label = label
    optimizer.zero_grad() 
    
    # Model forward 
    if "vit" not in args.basic_model:
        feats, logits, MCP_feat = model(b_data, concept_vectors, concept_means)
        logits_patch = None
    else:
        feats, logits, logits_patch, MCP_feat = model(b_data, concept_vectors, concept_means)
    if len(feats[0].shape) == 3:
        for i in range(len(feats)):
            feats[i] = feats[i].permute(0, 2, 1)

    if args.world_size > 1:
        for i in range(len(feats)):
            feats[i] = torch.cat(GatherLayer.apply(feats[i].contiguous()), dim = 0)
        MCP_feat = torch.cat(GatherLayer.apply(MCP_feat), dim = 0)
        logits = torch.cat(GatherLayer.apply(logits), dim = 0)
        if logits_patch is not None:
            logits_patch = torch.cat(GatherLayer.apply(logits_patch), dim = 0)
        gather_label = [torch.zeros_like(b_label) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_label, b_label)      
        b_label = torch.cat(gather_label, dim = 0)  

    # calculate loss
    if feats[0].shape[0] > 2:
        if "vit" not in args.basic_model:
            cka_loss = loss_funcs["CKA_loss"](feats)
        else:
            feat_cls = [feat[..., :1] for feat in feats]
            feat_patch = [feat[..., 1:] for feat in feats]
            cka_loss = (loss_funcs["CKA_loss"](feat_cls) + loss_funcs["CKA_loss"](feat_patch))
    else:
        cka_loss = torch.tensor(0).cuda()

    CCC_loss = loss_funcs["CCC_loss"](MCP_feat, b_label)
    ce_loss = loss_funcs["CE_loss"](logits, b_label)
    if logits_patch is not None:
        hard_label = torch.max(logits, dim = 1)[1]
        ce_loss = ce_loss + loss_funcs["CE_loss"](logits_patch, hard_label)
    loss = cka_loss + ce_loss + CCC_loss
    loss.backward()

    if torch.isnan(loss) or torch.isinf(loss):
        if args.global_rank in [-1, 0]:
            print("loss : ", loss)
            print("cka : ", cka_loss)
            print("ce : ", ce_loss)
            print("ccc : ", CCC_loss)
            torch.save({"MCP" : MCP_feat, 
                        "label" : b_label}, f"./error_feat_{args.case_name}.pkl")
        sys.exit()
        
    optimizer.step()
    
    losses = {
                "cka_loss" : cka_loss.detach(),
                "ce_loss" : ce_loss.detach(),
                "CCC_loss" : CCC_loss.detach(),
             }
    
    return losses

def test_one_step(model, data, label, args, concept_vecs = None, concept_means = None):
    if args.device_id != -1:
        b_data = data.to(args.device_id)
        b_label = label.to(args.device_id)
    else:
        b_data = data
        b_label = label
        
    # Model forward
    if "vit" not in args.basic_model:
        feats, logits, MCP_feat = model(b_data, concept_vecs, concept_means)
    else:
        # We utilize the logits from the MCP feature as the final result, which is based on these concept.
        feats, _, logits, MCP_feat = model(b_data, concept_vecs, concept_means)

    
    if args.world_size > 1:
        for i in range(len(feats)):
            feats[i] = torch.cat(GatherLayer.apply(feats[i].contiguous()), dim = 0)

        b_label = torch.cat(GatherLayer.apply(b_label), dim = 0)
        if logits is not None:
            logits = torch.cat(GatherLayer.apply(logits), dim = 0)

    predicted = None
    if logits is not None:
        _, predicted = torch.topk(logits.data, k = 5, dim = 1)
        predicted = predicted.detach()

    losses = {
    }
    return losses, feats, predicted

def get_optimizer(parameters, args):
    train_optimizer = None
    if args.optimizer == "adamw":
        train_optimizer = torch.optim.AdamW(parameters, lr = args.lr, weight_decay = args.weight_decay)

    return train_optimizer

def get_lr_scheduler(optimizer, steps, args):
    lr_scheduler = None
    if "lr_scheduler" in args:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = steps, eta_min = args.lr/args.lr_rate, last_epoch=-1)
    
    return lr_scheduler

def cal_cov_concept(Sum_As, Square_Sum_As, cov_xxs, cov_means):
    concept_vectors = [[]] * len(args.sel_layers)
    concept_means = [[]] * len(args.sel_layers)
    covs = []
    for i in range(len(args.sel_layers)):
        # calculate weighted covariance matrix
        cov, cov_mean = cal_cov(cov_xxs[i], cov_means[i], Sum_As[i])
        covs.append(cov)
        concept_means[i] = cov_mean
        # eigen decompose
        concept_vectors[i], concept_means[i] = cal_concept(cov, cov_mean)
    return concept_vectors, concept_means, covs

# =============================================================================
# Load data, load model (pretrain if needed), define loss function, define optimizer, 
# define learning rate scheduler (if needed), training and validation
# =============================================================================
def runs(args):
    # Load dataset ------------------------------------------------------------
    dataset, dataset_sizes, all_datasetsets = dataloader.load_data(args)
    # -------------------------------------------------------------------------
    
    # Define tensorboard for recording ----------------------------------------
    if args.global_rank in [-1, 0]:
        with open('{}/logging.txt'.format(args.dst), "a") as f:
            print('case_name : {}'.format(args.case_name), file = f)
            print("dataset : {}".format(args.dataset_name), file = f)
        writer = SummaryWriter(f'./logs/{args.case_name}/{args.basic_model}')
    # -------------------------------------------------------------------------
    
    # Load model (load pretrain if needed) ------------------------------------
    extra_kwargs = {}
    extra_kwargs["sel_layers"] = args.sel_layers
    extra_kwargs["drop_p"] = args.dropout_p
    model = load_model(args, **extra_kwargs)
    # -------------------------------------------------------------------------
    
    # Define loss -------------------------------------------------------------
    loss_funcs = {}
    loss_funcs["CCC_loss"] = CCC_with_drop_p(concept_per_layer = args.concept_per_layer)
    loss_funcs["CE_loss"] = nn.CrossEntropyLoss()
    loss_funcs["CKA_loss"] = CKA_loss_sampled(args.concept_per_layer)
    # -------------------------------------------------------------------------
    
    # Define optimizer --------------------------------------------------------
    train_optimizer = get_optimizer(model.parameters(), args)
    # -------------------------------------------------------------------------
    
    # Define learning rate scheduler ------------------------------------------
    lr_scheduler = get_lr_scheduler(train_optimizer, args.epoch * len(dataset["train"]), args)
    # -------------------------------------------------------------------------
        
    # Define Meters -------------------------------------------------------
    ACCMeters = AverageMeter()
    ACCMeters5 = AverageMeter()
    LOSSMeters = AverageMeter()

    max_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
    last_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
    # ---------------------------------------------------------------------
    
    concept_vectors = [[], [], [], []]
    concept_means = [[], [], [],[]]
    # ---------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------
    # Extract concept from pretrained -------------------------------------------------------------------------------------------
    cov_xxs = []
    cov_means = []
    Sum_As = []
    Square_Sum_As = []
    for layer_i in range(len(args.sel_layers)):
        cov_xxs.append(torch.zeros(args.concept_per_layer[layer_i], args.concept_cha[layer_i], args.concept_cha[layer_i], dtype = torch.float64).cuda(args.global_rank))
        cov_means.append(torch.zeros(args.concept_per_layer[layer_i], args.concept_cha[layer_i], 1, dtype = torch.float64).cuda(args.global_rank))
        Sum_As.append(torch.zeros(args.concept_per_layer[layer_i], dtype = torch.float64).cuda(args.global_rank))
        Square_Sum_As.append(torch.zeros(args.concept_per_layer[layer_i], dtype = torch.float64).cuda(args.global_rank))

    if args.global_rank in [-1, 0]:
        print("First epoch: Extract the concept vectors and concept means!!")
    train_transform = dataset["train"].dataset.transform
    val_transform = dataset["val"].dataset.transform
    dataset["train"].dataset.transform = val_transform
    model.train(False)
    with torch.no_grad():
        nb = len(dataset["sub_train"])
        pbar = enumerate(dataset["sub_train"])
        if args.global_rank in [-1, 0]:  
            pbar = tqdm.tqdm(pbar, total = nb)  # progress bar

        # Evaluate first time and Extract the concept vector
        for step, (data, label) in pbar:
            losses, feats, _ = test_one_step(model, data, label, args)
            if len(feats[0].shape) == 3:
                features = [feat[:, 1:].permute(0, 2, 1) for feat in feats]
            else:
                features = feats
            Sum_As, Square_Sum_As, cov_xxs, cov_means = cal_cov_component(features, Sum_As, Square_Sum_As, cov_xxs, cov_means, args)
        
        if args.world_size > 1:
            for i in range(len(features)):
                dist.all_reduce(Sum_As[i], op = dist.ReduceOp.SUM)
                dist.all_reduce(Square_Sum_As[i], op = dist.ReduceOp.SUM)
                dist.all_reduce(cov_xxs[i], op = dist.ReduceOp.SUM)
                dist.all_reduce(cov_means[i], op = dist.ReduceOp.SUM)
        
        concept_vectors, concept_means, _ = cal_cov_concept(Sum_As, Square_Sum_As, cov_xxs, cov_means)
        torch.cuda.empty_cache()
        
    if args.global_rank in [0, -1]:
        print("Finish extract concept and MCP features!!")

    # ---------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------
    # Train and Validation ---------------------------------------------------------------
    for epoch in range(1, args.epoch + 1):
        if args.global_rank in [-1, 0]:
            info_log('-' * 15, rank = args.global_rank, type = ["std", "log"], file = args.log)
            info_log('Epoch {}/{}'.format(epoch, args.epoch), rank = args.global_rank, type = ["std", "log"], file = args.log)
        
        torch.cuda.empty_cache()

        # ========================================================================================================================
        # ========================================================================================================================
        # train phase ============================================================================================================
        dataset["train"].dataset.transform = train_transform
        model.train(True)
        if args.global_rank != -1:
            dataset["train"].sampler.set_epoch(epoch)
            dataset["val"].sampler.set_epoch(epoch)
        loss_t = AverageMeter()
        loss_detail_t = {}
        nb = len(dataset["train"])
        pbar = enumerate(dataset["train"])
        if args.global_rank in [-1, 0]:
            pbar = tqdm.tqdm(pbar, total=nb)  # progress bar

        for step, (data, label) in pbar:
            losses = train_one_step(model = model, 
                                data = data,
                                label = label, 
                                loss_funcs = loss_funcs, 
                                optimizer = train_optimizer, 
                                args = args, 
                                concept_vectors = concept_vectors, 
                                concept_means = concept_means)
            
            # record losses
            loss = 0
            for key in losses.keys():
                loss_i = losses[key]
                dist.all_reduce(loss_i, op = dist.ReduceOp.SUM)
                loss_i = loss_i / args.world_size
                loss += loss_i
                if key not in loss_detail_t.keys():
                    loss_detail_t[key] = AverageMeter()

                if args.global_rank in [-1, 0]:  
                    loss_detail_t[key].update(loss_i, label.size(0) * args.world_size)
                losses[key] = loss_i.detach().item()
                
            if args.global_rank in [-1, 0]:
                loss_t.update(loss, label.size(0) * args.world_size)
                pbar.set_postfix(losses)
                
            lr_scheduler.step()

        if args.global_rank in [-1, 0]:
            writer.add_scalar('Loss/train', loss_t.avg, epoch)
            for key in loss_detail_t.keys():
                writer.add_scalar('{}/train'.format(key), loss_detail_t[key].avg, epoch)
            
        torch.cuda.empty_cache()
        # ========================================================================================================================
        # ========================================================================================================================
        # validation =============================================================================================================   
        cov_xxs = []
        cov_means = []
        Sum_As = []
        Square_Sum_As = []
        for layer_i in range(len(args.sel_layers)):
            cov_xxs.append(torch.zeros(args.concept_per_layer[layer_i], args.concept_cha[layer_i], args.concept_cha[layer_i], dtype = torch.float64).cuda(args.global_rank))
            cov_means.append(torch.zeros(args.concept_per_layer[layer_i], args.concept_cha[layer_i], 1, dtype = torch.float64).cuda(args.global_rank))
            Sum_As.append(torch.zeros(args.concept_per_layer[layer_i], dtype = torch.float64).cuda(args.global_rank))
            Square_Sum_As.append(torch.zeros(args.concept_per_layer[layer_i], dtype = torch.float64).cuda(args.global_rank))
        
        dataset["sub_train"].dataset.resample()
        dataset["train"].dataset.transform = val_transform
        model.train(False)
        for phase in ["sub_train", "val"]:
            correct_t = AverageMeter()
            correct_t5 = AverageMeter()

            loss_t = AverageMeter()
            loss_detail_t = {}
            
            with torch.no_grad():
                total_correct = 0
                total_count = 0
                nb = len(dataset[phase])
                pbar = enumerate(dataset[phase])
                if args.global_rank in [-1, 0]:  
                    pbar = tqdm.tqdm(pbar, total = nb)  # progress bar

                # Evaluate first time and Extract the concept vector
                for step, (data, label) in pbar:
                    if args.global_rank != -1:
                        b_label = label.cuda(args.global_rank)

                    losses, feats, predicted = test_one_step(model, data, label, args, concept_vectors, concept_means)
                    
                    if "train" in phase:
                        if len(feats[0].shape) == 3:
                            features = [feat[:, 1:].permute(0, 2, 1) for feat in feats]
                        else:
                            features = feats
                        Sum_As, Square_Sum_As, cov_xxs, cov_means = cal_cov_component(features, Sum_As, Square_Sum_As, cov_xxs, cov_means, args)

                    loss = 0
                    for key in losses.keys():
                        loss_i = losses[key]
                        dist.reduce(loss_i, 0, op = dist.ReduceOp.SUM)
                        loss_i = loss_i / args.world_size
                        loss += loss_i
                        if key not in loss_detail_t.keys():
                            loss_detail_t[key] = AverageMeter()

                        if args.global_rank in [-1, 0]:  
                            loss_detail_t[key].update(loss_i, data.size(0) * args.world_size)
                            
                    if args.global_rank in [-1, 0]: 
                        loss_t.update(loss, data.size(0) * args.world_size)

                    b_label_all = [torch.zeros_like(b_label) for _ in range(args.world_size)]
                    dist.all_gather(b_label_all, b_label)
                    b_label = torch.cat(b_label_all, dim = 0)
                    correct = (predicted[:, :1] == b_label.unsqueeze(1)).sum()
                    total_correct += correct
                    total_count += b_label.shape[0]
                    correct5 = (predicted[:, :5] == b_label.unsqueeze(1)).sum()
                    
                    if args.global_rank in [-1, 0]:  
                        correct_t.update(correct.item() / b_label.shape[0], b_label.shape[0])
                        correct_t5.update(correct5.item() / b_label.shape[0], b_label.shape[0])
                    
                if "train" in phase:
                    for i in range(len(args.sel_layers)):
                        dist.all_reduce(Sum_As[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(Square_Sum_As[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(cov_xxs[i], op = dist.ReduceOp.SUM)
                        dist.all_reduce(cov_means[i], op = dist.ReduceOp.SUM)

                    concept_vectors, concept_means, covs = cal_cov_concept(Sum_As, Square_Sum_As, cov_xxs, cov_means)
                      
            if args.global_rank in [-1, 0]:  
                # Recording loss and accuracy ---------------------------------
                if phase == "val":
                    writer.add_scalar('Loss/{}'.format(phase), loss_t.avg, epoch)
                    for key in losses.keys():
                        writer.add_scalar('{}/{}'.format(key, phase), loss_detail_t[key].avg, epoch)

                writer.add_scalar('Accuracy top1/{}'.format(phase), correct_t.avg, epoch)
                writer.add_scalar('Accuracy top5/{}'.format(phase), correct_t5.avg, epoch)
                # -------------------------------------------------------------
                
                # Save model --------------------------------------------------
                if "train" in phase:
                    phase = "train"
                if max_acc[phase].avg <= correct_t.avg:
                    last_acc[phase] = max_acc[phase]
                    max_acc[phase] = correct_t
                    
                    if phase == 'val':
                        ACCMeters = correct_t
                        LOSSMeters = loss_t
                        info_log('save')
                        optimizers_state_dict = train_optimizer.state_dict()
                        lr_state_dict = lr_scheduler.state_dict()
                            
                        save_data = {
                                        "Model" : model.state_dict(),
                                        "Epoch" : epoch,
                                        "Optimizer" : optimizers_state_dict,
                                        "Lr_scheduler" : lr_state_dict,
                                        "Best ACC" : max_acc[phase].avg,
                                        "concept_per_layer" : args.concept_per_layer,
                                        "concept_cha" : args.concept_cha
                                    }
                        torch.save(save_data, f"{args.dst}/best_model.pkl")
                        MCP_data = {
                                        "subset_idx" : dataset["sub_train"].dataset.indices,
                                        "concept_covs" : covs,
                                        "concept_means" : concept_means,
                                    }
                        torch.save(MCP_data, f"{args.dst}/MCP_data.pkl")

                optimizers_state_dict = train_optimizer.state_dict()
                lr_state_dict = lr_scheduler.state_dict()
                save_data = {
                                "Model" : model.state_dict(),
                                "Epoch" : epoch,
                                "Optimizer" : optimizers_state_dict,
                                "Lr_scheduler" : lr_state_dict,
                                "Best ACC" : max_acc[phase].avg,
                                "concept_per_layer" : args.concept_per_layer,
                                "concept_cha" : args.concept_cha
                            }
                torch.save(save_data, f"{args.dst}/last_model.pkl")
                MCP_data = {
                                "subset_idx" : dataset["sub_train"].dataset.indices,
                                "concept_covs" : covs,
                                "concept_means" : concept_means,
                            }
                torch.save(MCP_data, f"{args.dst}/last_MCP_data.pkl")
                # -------------------------------------------------------------
                info_log(f"case_name : {args.case_name}", rank = args.global_rank, type = ["std", "log"], file = args.log)        
                info_log(f"dataset : {args.dataset_name}", rank = args.global_rank, type = ["std", "log"], file = args.log)        
                info_log(f"Basic model : {args.basic_model}", rank = args.global_rank, type = ["std", "log"], file = args.log)        
                info_log("{} set loss : {:.6f}".format(phase, loss_t.avg), rank = args.global_rank, type = ["std", "log"], file = args.log)        
                for key in losses.keys():
                    info_log("{} set {} loss : {:.6f}".format(phase, key, loss_detail_t[key].avg), rank = args.global_rank, type = ["std", "log"], file = args.log)     
                info_log("{} set best acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.), rank = args.global_rank, type = ["std", "log"], file = args.log) 
                info_log("{} set top-1 acc : {:.6f}%".format(phase, correct_t.avg * 100.), rank = args.global_rank, type = ["std", "log"], file = args.log)  
                info_log("{} set top-5 acc : {:.6f}%".format(phase, correct_t5.avg * 100.), rank = args.global_rank, type = ["std", "log"], file = args.log)  
                info_log("{} last update : {:.6f}%".format(phase, (max_acc[phase].avg - last_acc[phase].avg) * 100.), rank = args.global_rank, type = ["std", "log"], file = args.log)
                info_log("-" * 10, rank = args.global_rank, type = ["std", "log"], file = args.log)
    # ---------------------------------------------------------------------
                
    # Show the best result ----------------------------------------------------
    info_log("Best acc(1) : {:.6f} acc(5) : {:.6f} loss : {:.6f}".format(ACCMeters.avg, ACCMeters5.avg, LOSSMeters.avg), rank = args.global_rank, type = ["std", "log"], file = args.log)

    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# Templet for recording values
# =============================================================================
class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count
        
if __name__ == '__main__':
    args = read_args()
    # Set DDP variables
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # check if it can run on gpu
    device_id = check_device(args.devices, args.train_batch_size, args.val_batch_size)
    args.train_total_batch_size = args.train_batch_size
    args.val_total_batch_size = args.val_batch_size
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device_id = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', device_id = torch.device(f"cuda:{args.local_rank}"), init_method='env://')  # distributed backend
        assert args.train_batch_size % args.world_size == 0, 'train_batch_size must be multiple of CUDA device count'
        args.train_batch_size = args.train_total_batch_size // args.world_size

    os.makedirs(f"{args.saved_dir}", exist_ok = True)
    args.dst = f"{args.saved_dir}/pkl/{args.case_name}/{args.basic_model}"
    args.log = '{}/logging.txt'.format(args.dst)
    if args.global_rank in [-1, 0]:
        first_time = False
        if not os.path.exists(f"{args.saved_dir}/pkl"):
            os.mkdir(f"{args.saved_dir}/pkl")
        if not os.path.exists(f"{args.saved_dir}/pkl/{args.case_name}/"):
            os.mkdir(f"{args.saved_dir}/pkl/{args.case_name}/")
        if not os.path.exists(f"{args.saved_dir}/pkl/{args.case_name}/{args.basic_model}"):
            first_time = True
            os.mkdir(f"{args.saved_dir}/pkl/{args.case_name}/{args.basic_model}")
        
        print(f"Args : {args}")
        if not first_time:
            response = input(f"The experiment already exist ({args.case_name}/{args.basic_model}). Are you sure you want replace it? (y/n)").lower()
            while response != 'y' and response != 'n':
                response = input(f"The experiment already exist ({args.case_name}/{args.basic_model}). Are you sure you want replace it? (y/n)").lower()
            if response == 'n':
                import sys
                sys.exit()
        
        with open(args.log, "w") as f:
            print(f"Args : {args}", file = f)
        # info_log(, args.global_rank, file = args.log)
        info_log(f"Save file to {args.dst}", args.global_rank, file = args.log)
        save_args(args)

        shutil.copy(src = os.path.join(os.getcwd(), __file__), dst = args.dst)
        os.makedirs(os.path.join(args.dst, "models"), exist_ok = True)
        if "resnet" in args.basic_model.lower():
            print(os.path.join(os.getcwd(), "models/ResNet.py"), os.path.join(args.dst, "models"))
            shutil.copy(src = os.path.join(os.getcwd(), "models/ResNet.py"), dst = os.path.join(args.dst, "models/ResNet.py"))
        elif "vit" in args.basic_model:
            shutil.copy(src = os.path.join(os.getcwd(), "models/vit.py"), dst = os.path.join(args.dst, "models/vit.py"))
        
        shutil.copytree(src = os.path.join(os.getcwd(), "utils"), dst = os.path.join(args.dst, "utils"), dirs_exist_ok = True)

    start = time.time()
    
    args.device_id = device_id
    runs(args)
    
    info_log("Train for {:.1f} hours".format((time.time() - start) / 3600), rank = args.global_rank, type = ["std", "log"], file = args.log)