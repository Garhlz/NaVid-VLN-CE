# run_robobrain.py

#!/usr/bin/env python3

import argparse
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from robobrain_agent import evaluate_agent

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    
    # --- 新增参数 ---
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Limit the number of episodes for a quick test.",
    )
    # --- 新增结束 ---

    args = parser.parse_args()
    run_exp(**vars(args))

# 注意修改函数签名，接收新的参数
def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, limit_episodes: int, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config = get_config(exp_config, opts)
            
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    
    dataset_split = dataset.get_splits(split_num)[split_id]

    # --- 新增逻辑 ---
    if limit_episodes is not None:
        print(f"--- SMOKE TEST: Limiting to first {limit_episodes} episodes. ---")
        dataset_split.episodes = dataset_split.episodes[:limit_episodes] # <--- 修改后的正确代码

    # 注意将新参数传递给函数
    evaluate_agent(config, split_id, dataset_split, model_path, result_path)

if __name__ == "__main__":
    main()