from difflib import SequenceMatcher
import argparse
import glob
import pandas as pd
import numpy as np
import torch
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', required = True, type = str)
    parser.add_argument('--model', default = "mcp_fc2_droput_l_2", type = str)
    parser.add_argument('--basic_model', default = "resnet50_relu", type = str)
    parser.add_argument('--dataset', default = "awa2", type = str, choices = ["awa2", "apascal", "lad"])
    parser.add_argument('--caption_model', default = "gpt-4o", type = str)
    parser.add_argument('--version', default = "v1", type = str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parser()
    if args.dataset == "awa2":
        class_paths = glob.glob("/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/train/*")
        class_names = [class_path.split("/")[-1] for class_path in class_paths]

        ## load concept sequence id
        ori_all_labels = pd.read_csv("/eva_data_4/bor/datasets/Animals_with_Attributes2/predicates.txt", header = None, sep = "\t")[1]

        # load class ground truth property from dataset (binary)
        case_property = np.loadtxt("/eva_data_4/bor/datasets/Animals_with_Attributes2/predicate-matrix-binary.txt")
        case_sequence = np.loadtxt("/eva_data_4/bor/datasets/Animals_with_Attributes2/classes.txt", dtype = str)

    elif args.dataset == "apascal":
        class_paths = glob.glob("/eva_data_4/bor/datasets/aPY/aPascal/train/*")
        class_names = [class_path.split("/")[-1] for class_path in class_paths]

        ## load concept sequence id
        ori_all_labels = pd.read_csv("/eva_data_4/bor/datasets/aPY/attribute_data/attribute_names.txt", header = None, sep = "\t")[0]

        # load class ground truth property from dataset (binary)
        case_property = torch.load("/eva_data_4/bor/datasets/aPY/aPascal/test_attribution.pkl")

    elif args.dataset == "lad":
        class_paths = glob.glob("/eva_data_4/bor/datasets/LAD/animal/train/*")
        class_names = [class_path.split("/")[-1][2:] for class_path in class_paths]
        attr_meta = pd.read_csv("/eva_data_4/bor/datasets/LAD/LAD_annotations/attribute_list.txt", header = None)
        ori_all_labels = attr_meta[attr_meta[0].str.contains('Attr_A')][1]

    save_path = f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model}/"
    found_concepts = pd.read_csv(f"{save_path}/concept caption_{args.version}.csv")
    found_concepts = found_concepts.fillna("None")

    print(found_concepts)

    csv_data = {"Layer_i":[],
                "Segment_i":[],
                "Concept_1":[],
                "Concept_2":[],
                "Concept_3":[]}
    
    for i in range(found_concepts.shape[0]):
        layer_i = found_concepts.iloc[i]["Layer_i"]
        segment_i = found_concepts.iloc[i]["Segment_i"]
        csv_data["Layer_i"].append(layer_i)
        csv_data["Segment_i"].append(segment_i)
        for concept_i in range(1, 4):
            concept = found_concepts.iloc[i][f"Concept_{concept_i}"]
            max_ratio = 0
            most_similar_label = ""
            for ori_label in ori_all_labels:
                ratio = SequenceMatcher(None, concept.split(":")[-1].strip().lower(), ori_label.split(":")[-1].strip().lower()).ratio()
                # print(ratio)
                if  ratio > max_ratio:
                    max_ratio = ratio
                    most_similar_label = ori_label.strip()

            if max_ratio > 0.9 or concept.split(":")[-1].strip().lower() in most_similar_label.strip():
                csv_data[f"Concept_{concept_i}"].append(most_similar_label)
            else:
                csv_data[f"Concept_{concept_i}"].append("None")

    df = pd.DataFrame(csv_data)
    df.to_csv(f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model}/concept caption_reform_{args.version}.csv", index = False)

