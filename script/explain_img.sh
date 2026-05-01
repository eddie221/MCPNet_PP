# AWA2
python -m explain_utils.explain_img --case_name AWA2_vit \
    --concept_per_layer 24 24 24 24 \
    --concept_cha 32 32 32 32 \
    --model MCPNet_pp \
    --basic_model vit_b_16 \
    --image_path "./sample/img.png" \
    --device 0 \
    --gt_class 44 \
    --saved_dir "." \
    --save_topk_prototypes 5 \
    --param_root "./"