#!/bin/bash

python train.py \
    --seq_pool_type mean \
    --vertex_pool_type max \
    --save_dir "results/template" \
    --rand_seed 5 \
    --conv_type graphconv \
    --dataset_has_fft \
    --model_dim 32 \
    --state_expansion_factor 16 \
    --local_conv_width 4 \
    --num_tgmamba_layers 2 \
    --num_vertices 19 \
    --rmsnorm \
    --edge_learner_layers 1 \
    --train_batch_size 256 \
    --val_batch_size 64 \
    --test_batch_size 64 \
    --num_workers 16 \
    --lr_init 5e-4 \
    --optimizer adam \
    --scheduler cosine \
    --num_epochs 100 \
    --patience 20 \
    --gpu_id 7 \
    --accumulate_grad_batches 1