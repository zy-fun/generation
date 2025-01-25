#!/bin/bash

python run.py \
    --task_name "exp" \
    --model "Transformer" \
    --test "True" \
    --load_model_path "checkpoints/2025-01-22_15-47-34/checkpoint.pth" \
    --data "shenzhen_20201104" \
    --use_subset "False" \
    --checkpoints "./checkpoints/" \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1 \
    --activation "gelu" \
    --use_pretrained "True" \
    --enc_emb 128 \
    --enc_dim 2 \
    --dec_dim 4 \
    --c_out 1 \
    --n_vocab 27411 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 10