
# AWA2 ViT-B
python -m classify_utils.cal_MCP_split_sub --case_name AWA2_vit_test3 \
    --device 0 --basic_model vit_b_16 \
    --concept_cha 32 32 32 32 --concept_per_layer 24 24 24 24 \
    --param_root "/eva_data_4/bor/MCPNet_redev/pkl"

python -m classify_utils.cal_acc_MCP_fc --case_name AWA2_vit_test3 \
    --basic_model vit_b_16 --device 0 \
    --concept_cha 32 32 32 32 --concept_per_layer 24 24 24 24 \
    --param_root "/eva_data_4/bor/MCPNet_redev/pkl"
