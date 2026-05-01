import pandas as pd
import argparse
from train_utils import basic_args, model_args

def arg_parser():
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument('--caption_model', default = "gpt-4o", type = str)
    parser.add_argument('--version', default = "v1", type = str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()

    concepts = []
    with open(f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model}/concept caption_{args.version}.txt", "r") as f:
        for line_i, line in enumerate(f.readlines()):
            if line_i in [0, 1, 2]:
                continue
            concepts.append(line.strip())

    csv_data = {"Layer_i":[],
                "Segment_i":[],
                "Concept_1":[],
                "Concept_2":[],
                "Concept_3":[]}

    args.concept_per_layer.insert(0, 0)
    concept_count = 0
    for layer_i in range(4):
        for concept_i in range(args.concept_per_layer[layer_i + 1]):
            concept_count = sum(args.concept_per_layer[:layer_i + 1]) + concept_i
            csv_data["Layer_i"].append(layer_i + 1)
            csv_data["Segment_i"].append(concept_i + 1)
            concept_split = concepts[concept_count].split(",")
            csv_data["Concept_1"].append(concept_split[0].strip().replace("+", " "))
            csv_data["Concept_2"].append(concept_split[1].strip().replace("+", " "))
            csv_data["Concept_3"].append(concept_split[2].strip().replace("+", " "))
            concept_count += 1

    df = pd.DataFrame(csv_data)
    df.to_csv(f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model}/concept caption_{args.version}.csv", index = False)

