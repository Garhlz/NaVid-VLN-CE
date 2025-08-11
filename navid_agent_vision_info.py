# 导入必要的库
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

# 从 NaVid 项目中导入相关的模块和常量
from navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from navid.conversation import conv_templates, SeparatorStyle
from navid.model.builder import load_pretrained_model
from navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# 主要的评估函数，保持不变
def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
    env = Env(config.TASK_CONFIG, dataset)
    agent = NaVid_Agent(model_path, result_path)
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
            action = agent.act(obs, info, env.current_episode.episode_id)
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}
            iter_step+=1
            obs = env.step(action)
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1
        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)


# 我们主要修改的 Agent 类
class NaVid_Agent(Agent):
    def __init__(self, model_path, result_path, require_map=True):
        
        print("Initialize NaVid Agent with Visual Memory Overlay")
        
        self.result_path = result_path
        self.require_map = require_map
        self.conv_mode = "vicuna_v1"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        print("Initialization Complete")

        # ======================= [核心修改 1/3] =======================
        # 恢复使用原始的 Prompt 模板。
        # 因为现在动作信息将通过视觉通道传递，我们不再需要修改文本输入。
        # 这确保了文本Prompt对于模型来说是完全“域内”和熟悉的。
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."
        # =============================================================

        self.history_rgb_tensor = None
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.reset()
    
    # ======================= [核心修改 2/3] =======================
    # 新增一个辅助函数，用于在图像上绘制上一步的动作文本。
    # def _draw_action_on_image(self, image: np.ndarray, last_action_id: int) -> np.ndarray:
    #     """
    #     在图像的左上角绘制上一步的动作。
    #     :param image: 原始 RGB 图像。
    #     :param last_action_id: 上一步执行的动作ID。
    #     :return: 绘制了文本的图像。
    #     """
    #     # 如果 last_action_id 是 None (通常在第一步)，则不绘制任何内容
    #     if last_action_id is None:
    #         return image
            
    #     action_map = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    #     action_text = action_map.get(last_action_id, "UNKNOWN")
        
    #     # 准备绘制文本的参数
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.6
    #     font_thickness = 2
    #     text_color = (255, 255, 0)  # 使用亮青色(Cyan)以示区分
    #     position = (10, 30) # 左上角位置
        
    #     # 复制图像以避免修改原始图像
    #     img_copy = image.copy()
        
    #     # 使用 OpenCV 的 putText 函数绘制文本
    #     cv2.putText(
    #         img_copy,
    #         f"Last Action: {action_text}",
    #         position,
    #         font,
    #         font_scale,
    #         text_color,
    #         font_thickness,
    #         lineType=cv2.LINE_AA
    #     )
    #     return img_copy
    # # =============================================================

    def _draw_action_on_image(self, image: np.ndarray, last_action_id: int) -> np.ndarray:
        """
        在图像底部中央绘制一个半透明的图标，表示上一步执行的动作。
        (已修正亮度问题)
        :param image: 原始 RGB 图像。
        :param last_action_id: 上一步执行的动作ID。
        :return: 绘制了图标的图像。
        """
        if last_action_id is None:
            return image
            
        # 复制一份原始图像，避免直接修改
        output_image = image.copy()
        h, w = image.shape[:2]

        # 1. 在一个单独的、全黑的“图标层”上绘制我们的图标
        icon_layer = np.zeros((h, w, 3), dtype=np.uint8)
        icon_color = (0, 255, 255)  # 亮青色
        thickness = 5

        if last_action_id == 1:  # FORWARD -> Arrow Up
            center_x = w // 2
            cv2.line(icon_layer, (center_x, h - 20), (center_x, h - 70), icon_color, thickness)
            cv2.line(icon_layer, (center_x, h - 70), (center_x - 15, h - 55), icon_color, thickness)
            cv2.line(icon_layer, (center_x, h - 70), (center_x + 15, h - 55), icon_color, thickness)
        elif last_action_id == 2:  # TURN_LEFT -> Arrow Counter-Clockwise
            center = (w // 2, h - 45)
            axes = (30, 30)
            cv2.ellipse(icon_layer, center, axes, 0, 90, 360, icon_color, thickness)
            triangle = np.array([ (center[0] - 30, center[1] - 10), (center[0] - 30, center[1] + 10), (center[0] - 40, center[1]) ])
            cv2.drawContours(icon_layer, [triangle], 0, icon_color, -1)
        elif last_action_id == 3:  # TURN_RIGHT -> Arrow Clockwise
            center = (w // 2, h - 45)
            axes = (30, 30)
            cv2.ellipse(icon_layer, center, axes, 0, -90, 180, icon_color, thickness)
            triangle = np.array([ (center[0] + 30, center[1] - 10), (center[0] + 30, center[1] + 10), (center[0] + 40, center[1]) ])
            cv2.drawContours(icon_layer, [triangle], 0, icon_color, -1)
        elif last_action_id == 0:  # STOP -> Square
            cv2.rectangle(icon_layer, (w // 2 - 25, h - 70), (w // 2 + 25, h - 20), icon_color, -1)

        # 2. 创建蒙版 (Mask)
        # 将图标层转换为灰度图
        gray_icon = cv2.cvtColor(icon_layer, cv2.COLOR_BGR2GRAY)
        # 使用阈值处理，生成一个二值的蒙版：图标区域是白色(255)，其余是黑色(0)
        _, mask = cv2.threshold(gray_icon, 1, 255, cv2.THRESH_BINARY)
        # 创建一个反向蒙版，用于提取背景
        mask_inv = cv2.bitwise_not(mask)

        # 3. 局部融合
        # 从原始图像中，使用反向蒙版挖出“背景”部分 (图标区域变黑)
        background = cv2.bitwise_and(output_image, output_image, mask=mask_inv)
        # 从图标层中，使用正向蒙版抠出“前景”（图标本身）
        foreground = cv2.bitwise_and(icon_layer, icon_layer, mask=mask)
        
        # 4. 将前景和背景直接相加，得到最终图像
        # 因为背景和前景的区域完全互补，所以可以直接用 cv2.add
        # 这样可以保证只有图标区域被修改，其他区域的亮度100%不变
        final_image = cv2.add(background, foreground)

        return final_image

    # process_images, predict_inference, extract_result, addtext 方法保持不变
    def process_images(self, rgb_list):
        start_img_index = 0
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        batch_image = np.asarray(rgb_list[start_img_index:])
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim = 0)
        return [self.history_rgb_tensor]

    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt
        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IMAGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        imgs = self.process_images(self.rgb_list)
        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def extract_result(self, output):
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None: return None, None
            return 1, float(match.group())
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None: return None, None
            return 2, float(match.group())
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None: return None, None
            return 3, float(match.group())
        return None, None

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2
        y_line = textY + 0 * textsize[1]
        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""
        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line
        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)
        return new_image


    def reset(self):
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))
                imageio.mimsave(output_video_path, self.topdown_map_list)
        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.first_forward = False


    def act(self, observations, info, episode_id):
        self.episode_id = episode_id

        # ======================= [核心修改 3/3] =======================
        # 这是整个方案最关键的修改点。
        
        # 1. 获取原始的 RGB 图像
        raw_rgb = observations["rgb"]
        
        # 2. 调用我们新增的辅助函数，将【上一步】的动作绘制到【当前】的图像上。
        # self.last_action 存储了上一步执行的动作ID。
        rgb_with_overlay = self._draw_action_on_image(raw_rgb, self.last_action)
        
        # 3. 将这张【修改后】的图像存入历史记录，用于模型的视觉输入。
        self.rgb_list.append(rgb_with_overlay)
        # =============================================================

        if self.require_map:
            # 注意：这里的 topdown_map 仍然使用原始的、未修改的 rgb 图像尺寸
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], raw_rgb.shape[0])
            output_im = np.concatenate((rgb_with_overlay, top_down_map), axis=1)

        # 首先检查 self.pending_action_list(待处理动作列表)是否为空
        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            # 更新 self.last_action，为下一次循环的绘制做准备
            self.last_action = temp_action
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}

        # 根据模板和当前的指令文本, 格式化出一个完整的提问
        navigation_qs = self.promt_template.format(observations["instruction"]["text"])
        
        # 将历史图像和问题一起打包发给大模型，获取模型生成的文本回答 navigation
        navigation = self.predict_inference(navigation_qs)
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)

        # 用正则表达式从模型返回的自然语言回答中，解析出结构化的动作指令
        action_index, num = self.extract_result(navigation[:-1])

        # 将结构化的指令分解成一连串的原子动作，存入self.pending_action_list
        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num/25))):
                self.pending_action_list.append(1)
        elif action_index == 2:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(2)
        elif action_index == 3:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(3)
        
        if action_index is None or len(self.pending_action_list)==0:
            self.pending_action_list.append(random.randint(1, 3))
        
        # 取出即将执行的第一个原子动作
        action_to_perform = self.pending_action_list.pop(0)
        
        # 更新 self.last_action，为下一次循环的绘制做准备
        self.last_action = action_to_perform
        
        return {"action": action_to_perform}