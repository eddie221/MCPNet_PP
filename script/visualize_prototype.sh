# AWA2 ViT-B
python -m vis_utils.find_topk_response --case_name AWA2_vit \
    --basic_model vit_b_16 --concept_cha 32 32 32 32 --concept_per_layer 24 24 24 24 \
    --device 0 --eigen_topk 1 \
    --param_root "./"

python -m vis_utils.find_topk_area --case_name AWA2_vit \
    --basic_model vit_b_16 --concept_cha 32 32 32 32 --concept_per_layer 24 24 24 24 \
    --topk 5 --eigen_topk 1 --individually --heatmap \
    --device 0 --param_root "./"
