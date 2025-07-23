#!/bin/bash

CUDA_VISIBLE_DEVICES=0,3 python train.py \
    --model_name "../models/llama3/sst2-sft-1000" \
    --file_path "./dataset/dgao/sst2/train_8_1000.jsonl" \
    --output_dir "results/llama/DGAO-sst2" \
    --task "sst2"