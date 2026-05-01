
# AWA2 on ViT-b-16
python -m torch.distributed.launch --nproc_per_node=8 --master_port 9568 train.py \
    --case_name AWA2_vit_test3 \
    --basic_model vit_b_16 \
    --device 0 1 2 3 4 5 6 7 \
    --dataset_name AWA2 \
    --concept_cha 32 32 32 32 \
    --concept_per_layer 24 24 24 24 \
    --optimizer adamw \
    --saved_dir "/eva_data_4/bor/MCPNet_redev" --dataloader load_data_train_val_classify_sub --num_samples_per_class 50 --rand_sub \
    --train_batch_size 192 --val_batch_size 32 --epoch 100