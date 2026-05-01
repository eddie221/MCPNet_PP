import base64
import requests
import os
import time
import glob
from dotenv import dotenv_values
import sys
import argparse
import pandas as pd
from train_utils import basic_args, model_args

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_concept_label(dataset_name = "awa2"):
    if dataset_name == "awa2":
        class_paths = glob.glob("/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/train/*")
        class_names = [class_path.split("/")[-1] for class_path in class_paths]
        all_labels = []
        with open("/eva_data_4/bor/datasets/Animals_with_Attributes2/predicates.txt", "r") as f:
            for line in f:
                all_labels.append(line.strip().split()[1])
    elif dataset_name == "apascal":
        class_paths = glob.glob("/eva_data_4/bor/datasets/aPY/aPascal/train/*")
        class_names = [class_path.split("/")[-1] for class_path in class_paths]
        all_labels = []
        with open("/eva_data_4/bor/datasets/aPY/attribute_data/attribute_names.txt", "r") as f:
            for line in f:
                all_labels.append(line.strip())
    elif dataset_name == "lad":
        class_paths = glob.glob("/eva_data_4/bor/datasets/LAD/animal/train/*")
        class_names = [class_path.split("/")[-1][2:] for class_path in class_paths]
        attr_meta = pd.read_csv("/eva_data_4/bor/datasets/LAD/LAD_annotations/attribute_list.txt", header = None)

        all_labels = []
        for i in range(attr_meta.shape[0]):
            if "Attr_A" in attr_meta.iloc[i][0]:
                all_labels.append(attr_meta.iloc[i][1])

    # "_1" => without class_names
    if "_1" not in args.version:
        all_labels = all_labels + class_names
        
    return all_labels

def arg_parser():
    parser = argparse.ArgumentParser(parents = [basic_args(), model_args()])
    parser.add_argument("--model", default = "MCPNet_pp", type = str)
    parser.add_argument('--caption_model', default = "gpt-4o", type = str)
    parser.add_argument('--dataset', default = "awa2", type = str, choices = ["awa2", "lad"])
    parser.add_argument('--start_layer', default = 1, type = int)
    parser.add_argument('--end_layer', default = 4, type = int)
    parser.add_argument('--concept_start_i', default = [1, 1, 1, 1], type = int, nargs = "+")
    parser.add_argument('--version', default = "v1", type = str)
    args = parser.parse_args()

    assert len(args.concept_per_layer) == len(args.concept_start_i), "Not match the layer of concept and start index !!"

    return args

if __name__ == "__main__":
    # OpenAI API Key
    config = dotenv_values(".env")
    openai_api = config["OPENAI_API_KEY"]

    args = arg_parser()

    concept_names = load_concept_label(args.dataset)

    args.save_path = f"./Concept Match/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/{args.caption_model.lower()}"
    os.makedirs(args.save_path, exist_ok = True)

    sys_prompt = "You are a helpful AI Assistant."
    # v1 (non-limit to dataset)
    if args.version == "v1":
        user_prompt = "You are an excellent investigator and will give precise and accurate answers. Your task is to identify objects or adjectives that are shared and exist in the first five images (not only exist in part of the images), and pick the most related description from the given concept list. The object or adjective concludes not only from the foreground but also from the background, but must appear simultaneously in the first five images, not only exist in part of the images. The last five patches correspond to the high-activation patches associated with the first five images. List three potential concepts shared in the first five images that refer to the high activation patches. The response should be detailed, brief, and easy to understand. The given concepts are: {}. If there is no properly matched concept, respond 'None'. Only response the result in format concept1, concept2, concept3."
    # v2 (limit to dataset)
    elif args.version == "v2":
        user_prompt = "You are an excellent investigator and will give precise and accurate answers. Firstly, you have to identify objects or adjectives that are shared and exist in the first five images and last five high activation patches (not only exist in part of the images) as much as possible. Then match these concepts with the given concept list. Finally, pick the concept that exists in most of the given images and patches. The object or adjective concludes not only from the foreground but also from the background, but must appear simultaneously in the first five images, not only exist in part of the images. The last five patches correspond to the high-activation patches associated with the first five images. The final result only lists three potential concepts shared in the first five images that refer to the high activation patches and don't duplicate. If the last five patches show a more consistent concept than the first five images, then the answer is based on the last five patches; vice versa. The response should be detailed, brief, and easy to understand. The given concepts are: {}. If there is no properly matched concept, respond 'None'. Only respond to the result in the format concept1, concept2, concept3."
                   
    user_prompt = user_prompt.format(", ".join(concept_names))

    with open(f"{args.save_path}/concept caption_{args.version}.txt", "w") as f:
        print(f"## System prompt : {sys_prompt}", file = f)
        print(f"## User prompt : {user_prompt}", file = f)
        print("", file = f)
        for layer_i in range(args.start_layer, args.end_layer + 1):
            for concept_i in range(args.concept_start_i[layer_i - 1], args.concept_per_layer[layer_i - 1] + 1):
                title = f"## Layer {layer_i} Concept {concept_i}"
                print(title)
                # Path to your image
                img_root = f"./find_topk_area_tmp/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}/5/ori_img"
                image_path1 = f"{img_root}/l{layer_i}_{concept_i}_1.png"
                image_path2 = f"{img_root}/l{layer_i}_{concept_i}_2.png"
                image_path3 = f"{img_root}/l{layer_i}_{concept_i}_3.png"
                image_path4 = f"{img_root}/l{layer_i}_{concept_i}_4.png"
                image_path5 = f"{img_root}/l{layer_i}_{concept_i}_5.png"

                patch_root = f"./extract_patch_tmp_split/{args.case_name}/{args.model.lower()}_{args.basic_model.lower()}"
                patch_path1 = f"{patch_root}/l{layer_i}_{concept_i}/l{layer_i}_{concept_i}_1.png"
                patch_path2 = f"{patch_root}/l{layer_i}_{concept_i}/l{layer_i}_{concept_i}_2.png"
                patch_path3 = f"{patch_root}/l{layer_i}_{concept_i}/l{layer_i}_{concept_i}_3.png"
                patch_path4 = f"{patch_root}/l{layer_i}_{concept_i}/l{layer_i}_{concept_i}_4.png"
                patch_path5 = f"{patch_root}/l{layer_i}_{concept_i}/l{layer_i}_{concept_i}_5.png"

                # Getting the base64 string
                concept_img1 = encode_image(image_path1)
                concept_img2 = encode_image(image_path2)
                concept_img3 = encode_image(image_path3)
                concept_img4 = encode_image(image_path4)
                concept_img5 = encode_image(image_path5)

                patch_img1 = encode_image(patch_path1)
                patch_img2 = encode_image(patch_path2)
                patch_img3 = encode_image(patch_path3)
                patch_img4 = encode_image(patch_path4)
                patch_img5 = encode_image(patch_path5)

                headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api}"
                }

                payload = {
                "model": args.caption_model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": user_prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{concept_img1}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{concept_img2}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{concept_img3}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{concept_img4}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{concept_img5}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{patch_img1}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{patch_img2}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{patch_img3}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{patch_img4}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{patch_img5}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 250
                }
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                try:
                    time.sleep(10)
                    response = response.json()
                    result = response['choices'][0]['message']['content'] + "," * max((2 - response['choices'][0]['message']['content'].count(",")), 0)
                    print(response)
                    print(result)
                    print(result, file = f)
                except:
                    print(response)
                    print(f"stop at {title}")
                    sys.exit()
                # print("", file = f)
            # print("", file = f)