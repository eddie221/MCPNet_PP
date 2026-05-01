
# AWA2 ViT-B
## Caption concept
python -m caption_utils.chat_gpt_match_concept \
    --case_name AWA2_vit \
    --basic_model vit_b_16 \
    --concept_per_layer 24 24 24 24

# change the format
python -m caption_utils.parse_caption --case_name AWA2_vit \
    --basic_model vit_b_16 \
    --concept_per_layer 24 24 24 24

# match the concept id from the dataset
python -m caption_utils.concept_match --case_name AWA2_vit \
    --basic_model vit_b_16 \
    --dataset awa2 \
    --concept_per_layer 24 24 24 24

# calculate the similarity
python -m caption_utils.eval_Caption_sim --case_name AWA2_vit \
    --basic_model vit_b_16 \
    --img_encoder ViT-L-14 \
    --pretrained_weight laion2b_s32b_b82k
