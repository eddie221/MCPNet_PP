import torch
import open_clip
import argparse
import pandas as pd
from PIL import Image
import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from train_utils import basic_args, model_args

def arg_parser():
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument("--sim_model", default = "clip", choices=["clip", "qwen"], type = str)
    parser.add_argument('--img_encoder', default = "ViT-B-32", type = str)
    parser.add_argument('--pretrained_weight', default = "laion2b_s34b_b79k", type = str)
    parser.add_argument('--caption_model', default = "gpt-4o", type = str)
    parser.add_argument('--version', default = "v1", type = str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()

    if args.sim_model == "clip":
        # load CLIP model ===================================================================================================
        model, _, preprocess = open_clip.create_model_and_transforms(args.img_encoder, pretrained = args.pretrained_weight, cache_dir = "/eva_data_4/bor/hf_model")
        model.eval()
        model.cuda()
        tokenizer = open_clip.get_tokenizer(args.img_encoder)
        # ===================================================================================================================
    elif args.sim_model == "qwen":
        # load Qwen model ===================================================================================================
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            cache_dir = "/eva_data_2/bor/hf_model",
            # quantization_config=quantization_config,
        )
        # model.save_pretrained("/eva_data_2/bor/hf_model_quantize")
        processor = AutoProcessor.from_pretrained(model_id)
        if args.batch != 1:
            processor.tokenizer.padding_side = "left"
        # ===================================================================================================================


    # load concepts =====================================================================================================
    caption_root = f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model}"
    caption_path = f"{caption_root}/concept caption_{args.version}.csv"
    concept_labels = pd.read_csv(caption_path, index_col = False, na_values = ["None", "none"])
    concept_labels = concept_labels.fillna("-")
    # ===================================================================================================================
    
    # load concept list ================================================================================================
    all_labels = []
    with open("/eva_data_4/bor/datasets/Animals_with_Attributes2/predicates.txt", "r") as f:
        for line in f:
            all_labels.append(line.strip().split()[1].lower())
    # ===================================================================================================================

    # calculate similarity ==============================================================================================
    concept_sims = {"concept" : [], "sim" : []}
    concept_sim_total = 0
    none_concept = 0
    with torch.no_grad():
        for concept_i in tqdm.tqdm(range(concept_labels.shape[0])):
            # concept_j = torch.randint(0, concept_labels.shape[0], (1,))
            concept_j = concept_i #concept_j[0].item()
            layer_i = concept_labels.iloc[concept_i]["Layer_i"]
            segment_i = concept_labels.iloc[concept_i]["Segment_i"]
            img_paths = [f"./find_topk_area_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/5/ori_img/l{layer_i}_{segment_i}_{i}.png" for i in range(1, 6)]
            patch_paths = [f"./extract_patch_tmp_split/{args.case_name}/{args.model.lower()}_{args.basic_model}/l{layer_i}_{segment_i}/l{layer_i}_{segment_i}_{i}.png" for i in range(1, 6)]
            # concept_texts = [concept_labels.iloc[concept_j]["Concept_1"].lower(), concept_labels.iloc[concept_j]["Concept_2"].lower(), concept_labels.iloc[concept_j]["Concept_3"].lower()]
            c1 = concept_labels.iloc[concept_j]["Concept_1"].lower()
            c2 = concept_labels.iloc[concept_j]["Concept_2"].lower()
            c3 = concept_labels.iloc[concept_j]["Concept_3"].lower()
            concept_texts = [f"a photo of {c1}", f"a photo of {c2}", f"a photo of {c3}"]
            # filter out the concept that doesn't exist in GT
            # for i in range(3):
            #     if concept_texts[i] not in all_labels:
            #         # print(concept_texts[i])
            #         concept_texts[i] = "-"
            # concept_texts = [concept_labels.iloc[concept_j]["Concept_1"].lower().replace("furn.", "furniture").replace("vert", "vertical").replace("horiz", "horizontal").replace("cyl", "cylindrical"), 
            #                  concept_labels.iloc[concept_j]["Concept_2"].lower().replace("furn.", "furniture").replace("vert", "vertical").replace("horiz", "horizontal").replace("cyl", "cylindrical"), 
            #                  concept_labels.iloc[concept_j]["Concept_3"].lower().replace("furn.", "furniture").replace("vert", "vertical").replace("horiz", "horizontal").replace("cyl", "cylindrical")]
            # print(concept_texts)
            none_idx = [i for i, concept in enumerate(concept_texts) if concept == "-"]
            none_mask = torch.ones(len(concept_texts)).cuda()
            none_mask[none_idx] = 0
            none_mask = none_mask.unsqueeze(0).repeat(len(img_paths), 1)

            img_feats = torch.stack([preprocess(Image.open(img_path)) for img_path in img_paths], dim = 0).cuda()
            patch_feats = torch.stack([preprocess(Image.open(patch_path)) for patch_path in patch_paths], dim = 0).cuda()
            concept_texts_idx = tokenizer(concept_texts).cuda()
            img_feats = model.encode_image(img_feats)
            patch_feats = model.encode_image(patch_feats)
            concept_texts_feats = model.encode_text(concept_texts_idx)

            img_feats /= img_feats.norm(dim = -1, keepdim = True)
            patch_feats /= patch_feats.norm(dim = -1, keepdim = True)
            concept_texts_feats /= concept_texts_feats.norm(dim = -1, keepdim = True)

            img_sim = torch.sum(img_feats[:, None, :] * concept_texts_feats[None, :, :], dim = -1)
            patch_sim = torch.sum(patch_feats[:, None, :] * concept_texts_feats[None, :, :], dim = -1)

            # get the most similar image / patch to the concept
            all_sim = torch.max(torch.stack([img_sim, patch_sim], dim = -1), dim = -1)[0]

            # select the similarity from the most similar concept
            all_sim_topk, _ = torch.topk(all_sim, dim = 1, k = 1)
            all_sim = all_sim_topk

            # average all five pairs
            if torch.sum(none_mask) == 0:
                concept_sim = torch.tensor(0)
                none_concept += 1
            else:
                concept_sim = torch.sum(all_sim * none_mask) / torch.sum(none_mask)

            concept_sims["concept"].append(f"L{layer_i}C{segment_i}")
            concept_sims["sim"].append(concept_sim.detach().cpu().item())
            # concept_sims[f"L{layer_i}C{segment_i}"] = concept_sim.detach().cpu().item()
            if torch.sum(none_mask) != 0:
                concept_sim_total += concept_sim
    
    print("Not match concept : ", none_concept)
    print((concept_sim_total / (concept_labels.shape[0] - none_concept)))
    concept_sims["concept"].append("avg")
    concept_sims["sim"].append((concept_sim_total / (concept_labels.shape[0] - none_concept)).detach().cpu().item())
    concept_sims = pd.DataFrame(concept_sims)
    print(f"{caption_root}/concept_sim_{args.img_encoder}_{args.version}.csv")
    concept_sims.to_csv(f"{caption_root}/concept_sim_{args.img_encoder}_{args.version}.csv", index = False)
    # ===================================================================================================================