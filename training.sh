#!/bin/bash

# Define the Python script name
PYTHON_SCRIPT="train.py"

# Run with different seq_pool_type and vertex_pool_type combinations
for seq_pool in "last" "mean" "max"
do
    for vertex_pool in "mean" "max"
    do
        echo "Running with seq_pool_type=$seq_pool and vertex_pool_type=$vertex_pool"
        python $PYTHON_SCRIPT \
            --seq_pool_type $seq_pool \
            --vertex_pool_type $vertex_pool \
            --save_dir "TGMamba/results/${seq_pool}_${vertex_pool}" \
            --rand_seed 42 \
            --dataset_has_fft True \
            --conv_type gcnconv \
            --model_dim 32 \
            --state_expansion_factor 16 \
            --local_conv_width 4 \
            --num_vertices 19 \
            --train_batch_size 1024 \
            --val_batch_size 256 \
            --test_batch_size 256 \
            --num_workers 16 \
            --lr_init 1e-3 \
            --optimizer adam \
            --scheduler cosine \
            --num_epochs 100 \
            --patience 5 \
            --gpu_id 5 \
            --accumulate_grad_batches 1
    done
done