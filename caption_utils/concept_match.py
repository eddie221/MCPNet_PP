import pandas as pd
import glob
import numpy as np
import sys
import argparse
import torch
import torchvision
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

        return label, image_path

def arg_parser():
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
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
        ori_all_concepts = pd.read_csv("/eva_data_4/bor/datasets/Animals_with_Attributes2/predicates.txt", header = None, sep = "\t")[1]
        
        # load class ground truth property from dataset (binary)
        case_property = np.loadtxt("/eva_data_4/bor/datasets/Animals_with_Attributes2/predicate-matrix-binary.txt")
        case_sequence = np.loadtxt("/eva_data_4/bor/datasets/Animals_with_Attributes2/classes.txt", dtype = str)

    elif args.dataset == "apascal":
        class_paths = glob.glob("/eva_data_4/bor/datasets/aPY/aPascal/train/*")
        class_names = [class_path.split("/")[-1] for class_path in class_paths]

        ## load concept sequence id
        ori_all_concepts = pd.read_csv("/eva_data_4/bor/datasets/aPY/attribute_data/attribute_names.txt", header = None, sep = "\t")[0]

        # load class ground truth property from dataset (binary)
        case_property = torch.load("/eva_data_4/bor/datasets/aPY/aPascal/test_attribution.pkl")

    elif args.dataset == "lad":
        class_paths = glob.glob("/eva_data_4/bor/datasets/LAD/animal/train/*")
        class_names = [class_path.split("/")[-1][2:] for class_path in class_paths]
        attr_meta = pd.read_csv("/eva_data_4/bor/datasets/LAD/LAD_annotations/attribute_list.txt", header = None)
        ori_all_concepts = attr_meta[attr_meta[0].str.contains('Attr_A')][1]
        attr_len = ori_all_concepts.shape[0]

    ori_all_concepts = [ori_all_label.lower().strip() for ori_all_label in ori_all_concepts]

    rest_labels = list(ori_all_concepts.copy())# + class_names
    rest_labels = [rest_label.lower().strip() for rest_label in rest_labels]
    all_concept_mask = np.zeros(len(ori_all_concepts))
    ori_total_concept_num = len(ori_all_concepts)

    save_path = f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model}/"
    if args.dataset == "lad":
        found_concepts = pd.read_csv(f"{save_path}/concept caption_reform_{args.version}.csv")
    else:
        found_concepts = pd.read_csv(f"{save_path}/concept caption_{args.version}.csv")

        
    found_concepts = found_concepts.fillna("None")
    overlapped_concepts = []
    for i, row in found_concepts.iterrows():
        for concept_i in range(1, 4):
            if row[f"Concept_{concept_i}"].lower() in rest_labels:
                rest_labels.remove(row[f"Concept_{concept_i}"].lower())
                overlapped_concepts.append(row[f"Concept_{concept_i}"].lower())
                concept_idx = ori_all_concepts.index(row[f"Concept_{concept_i}"].lower())
                all_concept_mask[concept_idx] = 1
            
            # deal with the case if separator (e.g. and, /) in the concept
            for sep in ["and", "/"]:
                if sep in row[f"Concept_{concept_i}"].lower():
                    concepts = row[f"Concept_{concept_i}"].lower().split(sep)
                    concept1 = concepts[0].strip()
                    concept2 = concepts[1].strip()
                    for concept in [concept1, concept2]:
                        if concept in rest_labels:
                            rest_labels.remove(concept)
                            overlapped_concepts.append(concept)
                            concept_idx = ori_all_concepts.index(concept.lower())
                            all_concept_mask[concept_idx] = 1

    # filter out the found concept
    filtered_labels = [label for mask, label in zip(all_concept_mask, ori_all_concepts) if mask == 1]
    print("filtered_labels : ", len(filtered_labels))
    print(filtered_labels)

    # store the evaluation gt label
    if args.dataset == "awa2":
        ### get original class idx
        class_map = {class_seq[1]: int(class_seq[0]) for class_seq in case_sequence}
        ### sort by class name
        class_map = {k: v - 1 for k, v in sorted(class_map.items(), key=lambda item: item[0])}
        class_idx = list(class_map.values())
        case_property = case_property[class_idx]
        filtered_case_property = case_property[:, all_concept_mask == 1]
        filtered_case_property = filtered_case_property.astype(np.int32)
        np.savetxt(f"{save_path}/AWA2_filtered_concept_binary_{args.version}.txt", filtered_case_property, fmt = '%i')
    elif args.dataset == "apascal":
        for key in case_property.keys():
            case_property[key] =  case_property[key][all_concept_mask == 1]
        torch.save(case_property, f"{save_path}/Filtered_concept_test_{args.version}.pkl")
    elif args.dataset == "lad":
        val_meta = pd.read_csv("/eva_data_4/bor/datasets/LAD/LAD_annotations/attributes.txt", header = None)
        img_paths = sorted(glob.glob("/eva_data_4/bor/datasets/LAD/animal/val/*/*"))
        img_paths = [img_path.replace("/eva_data_4/bor/datasets/LAD/animal/val", "images") for img_path in img_paths]
        gt_label = np.zeros((len(img_paths), attr_len))
        val_meta = val_meta[val_meta[0].str.contains('Label_A')]
        
        all_val_data = {"img_path" : [],
                        "concept_label" : []}
        for i in range(val_meta.shape[0]):
            img_path = val_meta.iloc[i][1].strip()
            index = img_paths.index(img_path)

            concept_label = val_meta.iloc[i][2].strip()[2: -2]
            concept_label = concept_label.split("  ")[:attr_len]
            concept_label = [int(label) for label in concept_label]
            gt_label[index] = np.array(concept_label)
            all_val_data["img_path"].append(img_path)
            all_val_data["concept_label"].append(concept_label)
        np.save(f"{save_path}/Val_metadata_{args.version}.npy", all_val_data)
        np.save(f"{save_path}/Val_data_{args.version}.npy", gt_label)


    # match found concept with ground truth concept
    found_concepts_match = {}
    for i, row in found_concepts.iterrows():
        layer_i = row["Layer_i"]
        segment_i = row["Segment_i"]
        found_concepts_match[f"L{layer_i}C{segment_i}"] = []
        for concept_i in range(1, 4):
            if row[f"Concept_{concept_i}"].lower() in overlapped_concepts:
                found_concepts_match[f"L{layer_i}C{segment_i}"].append(filtered_labels.index(row[f"Concept_{concept_i}"].lower()))
        
        for sep in ["and", "/"]:
            if sep in row[f"Concept_{concept_i}"].lower():
                concepts = row[f"Concept_{concept_i}"].lower().split(sep)
                concept1 = concepts[0].strip()
                concept2 = concepts[1].strip()
                for concept in [concept1, concept2]:
                    if concept in overlapped_concepts:
                        found_concepts_match[f"L{layer_i}C{segment_i}"].append(filtered_labels.index(concept))
                        
    if args.dataset == "awa2":
        np.save(f"{save_path}/AWA2_concept_match_{args.version}.npy", found_concepts_match)
        np.savetxt(f"{save_path}/AWA2_filtered_label_{args.version}.txt", filtered_labels, fmt = "%s")
    elif args.dataset == "apascal":
        np.save(f"{save_path}/aPascal_concept_match_{args.version}.npy", found_concepts_match)
        np.savetxt(f"{save_path}/aPascal_filtered_label_{args.version}.txt", filtered_labels, fmt = "%s")
    elif args.dataset == "lad":
        np.save(f"{save_path}/LAD_concept_match_{args.version}.npy", found_concepts_match)
        np.savetxt(f"{save_path}/LAD_filtered_label_{args.version}.txt", filtered_labels, fmt = "%s")
    print("Finish")