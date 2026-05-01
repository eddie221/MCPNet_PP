
# AWA2 ViT-B
python -m vis_utils.extract_patch --case_name AWA2_vit \
    --basic_model vit_b_16 \
    --concept_cha 32 32 32 32 --concept_per_layer 24 24 24 24 \
    --topk 25 --device 0 --scale 2
