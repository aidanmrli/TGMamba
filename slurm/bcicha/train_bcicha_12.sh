#!/bin/bash

python /home/amli/TGMamba/train.py \
    --save_dir /home/amli/TGMamba/results/template \
    --rand_seed 123 \
    --dataset bcicha \
    --dataset_has_fft \
    --subject 12 \
    --conv_type graphconv \
    --local_conv_width 4 \
    --num_epochs 1000 \
    --patience 150 \
    --gpu_id 5 \
    --attn_softmax_temp 0.4994 \
    --attn_threshold 0.047702 \
    --edge_learner_attention --edge_learner_layers 1 --edge_learner_time_varying --lr_init 0.00007167 --model_dim 16 --num_tgmamba_layers 1 --seq_pool_type last --state_expansion_factor 32 --vertex_pool_type mean --weight_decay 0.010016 --rmsnorm --train_batch_size 60 --val_batch_size 60 --test_batch_size 60 --num_workers 12 --optimizer_name adamw --scheduler cosine