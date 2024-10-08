#!/bin/bash

python /home/amli/TGMamba/train.py \
    --save_dir "/home/amli/TGMamba/results/template" \
    --rand_seed 123 \
    --dataset 'tuhz' \
    --dataset_has_fft \
    --seq_pool_type mean \
    --vertex_pool_type max \
    --conv_type graphconv \
    --model_dim 50 \
    --state_expansion_factor 48 \
    --local_conv_width 4 \
    --num_tgmamba_layers 1 \
    --rmsnorm \
    --edge_learner_attention \
    --edge_learner_layers 1 \
    --edge_learner_time_varying \
    --attn_softmax_temp 0.5 \
    --attn_threshold 0.2 \
    --train_batch_size 256 \
    --val_batch_size 64 \
    --test_batch_size 64 \
    --num_workers 12 \
    --lr_init 0.00015 \
    --weight_decay 0.3 \
    --optimizer_name adamw \
    --scheduler cosine \
    --num_epochs 100 \
    --patience 20 \
    --gpu_id 0 \