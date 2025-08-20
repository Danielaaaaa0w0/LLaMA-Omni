#!/bin/bash

# ROOT=$1

# VOCODER_CKPT=vocoder/g_00500000
# VOCODER_CFG=vocoder/config.json

python omni_speech/infer/infer.py \
    --model-path /mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/saves3/checkpoint-139270 \
    --question-file /mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/mandarin_100h_json/dev_mandarin_100h.json \
    --answer-file /mnt/md0/user_yuze0w0/LLAMA_tw-zh/LLaMA-Omni/mandarin_100h_json/answer_ckpt139270_dev.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \

# python omni_speech/infer/convert_jsonl_to_txt.py $ROOT/answer.json $ROOT/answer.unit
# python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
#     --in-code-file $ROOT/answer.unit \
#     --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
#     --results-path $ROOT/answer_wav/ --dur-prediction

# bash omni_speech/infer/run.sh omni_speech/infer/examples
