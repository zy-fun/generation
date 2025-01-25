#!/bin/bash

python run.py \
    --task_name "exp" \
    --model "VAETransformer" \
    --data "shenzhen_20201104" \
    --checkpoints "./checkpoints/" \
    --d_model 64 \
    --d_ff 256 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1 \
    --activation "gelu" \
    --use_pretrained True \
    --enc_emb 128 \
    --enc_dim 2 \
    --dec_dim 4 \
    --c_out 1 \
    --n_vocab 27411 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --load_model_path "checkpoints/VAETransformer/checkpoint.pth" \
    --test >> log.txt