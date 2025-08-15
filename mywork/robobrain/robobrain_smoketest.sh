#!/bin/bash

MODEL_PATH="/data/model_zoo/RoboBrain2.0-7B" 

CHUNKS=1 # 冒烟测试只需要处理一个块
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
SAVE_PATH="tmp/robobrain_on_r2r_smoke_test" # 使用专用的测试结果路径

echo "Running smoke test on GPU 0 for 2 episodes..."
CUDA_VISIBLE_DEVICES=0 python run_smoketest.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id 0 \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH \
    --limit-episodes 2  # <-- 使用我们新增的参数，只跑2个episodes

echo "Smoke test finished."