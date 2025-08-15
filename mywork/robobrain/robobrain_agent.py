import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
import torch
import cv2
import imageio
from habitat.utils.visualizations import maps
import random

# from inference import SimpleInference
from inference_navi import SimpleInference

def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
 
    env = Env(config.TASK_CONFIG, dataset)

    agent = RoboBrain_Nav_Agent(model_path, result_path)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0
    
      
    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id)):
        obs = env.reset()
        iter_step = 0
        agent.reset()

         
        continuse_rotation_count = 0
        last_dtg = 999
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            # 输入观测, agent思考之后, 返回一个决策action
            action = agent.act(obs, info, env.current_episode.episode_id)

            # 防卡死, 如果连续旋转次数过多或总步数超限，就会强制执行STOP动作
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            
            iter_step+=1
            obs = env.step(action)
            # 执行动作后，获取新的观测信息
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1



        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)


class RoboBrain_Nav_Agent(Agent):
    def __init__(self, model_path, result_path, require_map=True):
        print("Initializing RoboBrain Navigation Agent...")
        self.result_path = result_path
        self.require_map = require_map
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        
        # 加载 RoboBrain 模型
        self.robobrain_model = SimpleInference(model_path)

        # --- 关键修改：在这里初始化所有属性 ---
        self.inference_times = []
        self.rgb_list = []
        self.pending_action_list = []
        self.topdown_map_list = []
        self.history_rgb_tensor = None
        self.transformation_list = []
        self.last_action = None
        self.count_id = 0
        self.count_stop = 0
        self.first_forward = False
        self.episode_id = None # 初始化 episode_id
        # --- 修改结束 ---

        print("RoboBrain Agent Initialization Complete")

    def reset(self):
        # 在每个 episode 结束时，计算并打印平均推理时间
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            print(f"--- Episode {self.episode_id} Summary ---") # 使用 self.episode_id
            print(f"Average inference time: {avg_time:.4f} seconds over {len(self.inference_times)} decisions.")
            print(f"------------------------------------")

        # 保存视频的逻辑
        if self.require_map and self.topdown_map_list:
            output_video_path = os.path.join(self.result_path, "video", f"{self.episode_id}.gif")
            imageio.mimsave(output_video_path, self.topdown_map_list)

        # --- 关键修改：这里只负责清空/重置，不再负责创建 ---
        self.inference_times.clear()
        self.rgb_list.clear()
        self.pending_action_list.clear()
        self.topdown_map_list.clear()
        self.transformation_list.clear()
        # --- 修改结束 ---
        
        self.history_rgb_tensor = None
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.first_forward = False

    # ! change 增加了图片缩放的逻辑
    # 新增一个辅助函数，因为 RoboBrain 的 inference 脚本需要文件路径
    def _save_temp_frame(self, rgb_array, max_dim=256): # 增加一个参数来控制尺寸
        """Saves and resizes the current RGB frame to a temporary file."""
        temp_path = "/tmp/robobrain_nav_frame.jpg"
        
        # 将 Habitat 的 RGB 格式 (H, W, C) numpy 数组转换为 OpenCV 的 BGR 格式
        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        # --- 新增：在这里调用缩放逻辑 ---
        h, w, _ = bgr_image.shape
        scale = max_dim / max(h, w)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(bgr_image, (new_w, new_h))
            cv2.imwrite(temp_path, resized)
        else:
            cv2.imwrite(temp_path, bgr_image)
        # --- 新增结束 ---

        return temp_path

    def act(self, observations, info, episode_id):
        # --- 在这里加入打印语句 ---
        # rgb_image_array = observations["rgb"]
        # print(f"--- Habitat Image Resolution (Height, Width, Channels): {rgb_image_array.shape} ---")
        # --- 打印语句结束 ---

        # a. 累积并保存所有历史 RGB 帧的路径
        #    注意：这里我们保存的是帧本身，而不是路径，后面统一处理
        self.rgb_list.append(observations["rgb"])

        # b. 优先执行待办动作列表
        if len(self.pending_action_list) != 0:
            return {"action": self.pending_action_list.pop(0)}

        # c. 当待办列表为空时，调用 RoboBrain 进行新决策
        instruction = observations["instruction"]["text"]
    
    
        # 1. 准备包含历史和当前的完整视觉输入
        
        # 为了效率和避免序列过长，只取最近的几帧作为历史
        # ! change: 这里假设最多使用最近3帧
        MAX_HISTORY_FRAMES = 1
        start_index = max(0, len(self.rgb_list) - 1 - MAX_HISTORY_FRAMES)
        
        # 从 self.rgb_list 中选出历史帧和当前帧
        frames_to_process = self.rgb_list[start_index:]
        
        # 将这些帧（numpy数组）保存为临时文件，并获取它们的路径列表
        # (这是一个简化处理，更高效的方式是直接传递图像对象，但这需要修改inference.py)
        image_paths = [self._save_temp_frame(frame) for frame in frames_to_process]
        
        
        # 调用 RoboBrain 模型进行推理
        pred = self.robobrain_model.inference(
            text=instruction,
            image=image_paths, # 关键修改：传入包含多张历史图片的列表
            task="navigation",
            enable_thinking=False,
        )
        navigation_output = pred['answer']
        
        if 'inference_time_seconds' in pred:
            self.inference_times.append(pred['inference_time_seconds'])
    

        # d. 解析模型输出 (这部分逻辑不变)
        action_index, num = self.extract_result(navigation_output)

        # e. 分解高级指令为原子动作 (这部分逻辑不变)
        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1 and num is not None:
            # 根据模型输出的距离，决定走几步
            steps = max(1, int(num / 25)) # 至少走一步
            for _ in range(steps):
                self.pending_action_list.append(1)
        elif action_index in [2, 3] and num is not None:
            # 根据模型输出的角度，决定转几步
            steps = max(1, int(num / 30)) # 至少转一次
            action_code = 2 if action_index == 2 else 3
            for _ in range(steps):
                self.pending_action_list.append(action_code)
        else: 
            print(f"Warning: Could not parse model output: '{navigation_output}'. Turning left as default.")
            self.pending_action_list.append(2)

        # 确保列表不为空
        if not self.pending_action_list:
            self.pending_action_list.append(2) # 添加一个默认动作以防万一

        return {"action": self.pending_action_list.pop(0)}

    def extract_result(self, output):
        # 这个函数完全可以从 navid_agent.py 中复制过来，很通用
        # id: 0-stop, 1-forward, 2-left, 3-right
        output = output.lower()
        if "stop" in output:
            return 0, None
        
        value = None
        match = re.search(r'-?\d+', output)
        if match:
            value = float(match.group())

        if "forward" in output:
            return 1, value
        elif "left" in output:
            return 2, value
        elif "right" in output:
            return 3, value
        
        return None, None

