#!/bin/bash

MODEL_PATH="/data/model_zoo/RoboBrain2.0-7B" 

# 其他配置可以保持不变
CHUNKS=8
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
SAVE_PATH="tmp/robobrain_on_r2r" 

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( IDX % 8 ))
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python run_robobrain.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH &
done

wait