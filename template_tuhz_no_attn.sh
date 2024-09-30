#!/bin/bash

python /home/amli/TGMamba/train.py \
    --save_dir "/home/amli/TGMamba/results/template" \
    --rand_seed 45 \
    --dataset 'dodh' \
    --dataset_has_fft \
    --seq_pool_type max \
    --vertex_pool_type mean \
    --conv_type graphconv \
    --model_dim 32 \
    --state_expansion_factor 48 \
    --local_conv_width 4 \
    --num_tgmamba_layers 1 \
    --rmsnorm \
    --edge_learner_layers 3 \
    --edge_learner_time_varying \
    --init_skip_param 0.528 \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --test_batch_size 64 \
    --num_workers 12 \
    --lr_init 0.0002 \
    --weight_decay 0.332 \
    --optimizer_name adamw \
    --scheduler cosine \
    --num_epochs 200 \
    --patience 50 \
    --gpu_id 2 \