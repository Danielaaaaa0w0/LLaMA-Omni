#!/bin/bash

# ROOT=$1

# VOCODER_CKPT=vocoder/g_00500000
# VOCODER_CFG=vocoder/config.json

python omni_speech/train/stage1.py \
    --model-path Llama-3.2-1B-Instruct \
    --question-file /mnt/md0/user_yuze0w0/dataset/mixed_data/train_mix_shuffled.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \
