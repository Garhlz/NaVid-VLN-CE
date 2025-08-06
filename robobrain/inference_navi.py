import os
import re
import cv2
import time # 导入 time 模块用于计时
from typing import Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

class SimpleInference:
    """
    A class for performing inference using Hugging Face models.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain2.0-7B"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier (default: "BAAI/RoboBrain2.0-7B")
        """
        print("Loading Checkpoint ...")

        # 使用量化之后速度反而变慢了...
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True
        # )

        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_id, 
        #     quantization_config=quantization_config, # 使用新的参数
        #     device_map="auto"
        # )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            # torch_dtype=torch.float16, # 使用半精度以节省显存
            torch_dtype = "auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
    # ! change 修改图片大小
    def _resize_image(self, image_path, output_path, max_dim=256):
        """
        Resizes an image to a maximum dimension while maintaining aspect ratio.
        This is a helper method to prevent CUDA out of memory issues with large images.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Unable to read image at {image_path}. Skipping resize.")
                return image_path
                
            h, w, _ = img.shape
            scale = max_dim / max(h, w)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h))
                cv2.imwrite(output_path, resized)
                print(f"Image resized to {new_w}x{new_h} and saved to {output_path}")
                return output_path
            return image_path
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image_path
        
    def inference(self, text:str, image: Union[list,str], task="general", enable_thinking=False):
        """
        Perform inference with text and images input.
        """
        if isinstance(image, str):
            image = [image]

        assert task in ["general", "navigation"], f"Invalid task type: {task}. Supported tasks are 'general', 'navigation'."

        if task == "navigation":
            # print("Navigation task detected. Applying few-shot prompt.")
            user_instruction = text 

            # 似乎和指令的长度关系不是很大...
            navigation_template = """You are a navigation robot agent controlling a real robot in a house. Your task is to follow the user's instruction based on your current camera view and a history of recent views.
--- RULES ---
1. Your ENTIRE response MUST be ONE of the four following action commands.
2. Do NOT output any other explanations, conversations, or text. Just the single action command.
3. The distance for 'move forward' MUST be in centimeters (cm).
4. The angle for turns MUST be in degrees.

--- AVAILABLE ACTIONS & FORMATS ---
* `The next action is move forward 25 cm`
* `The next action is turn left 30 degree`
* `The next action is turn right 30 degree`
* `The next action is stop`

--- EXAMPLES ---
Example 1:
User Instruction: "go forward towards the door"
*[Image shows a door far ahead]*
Your Output: The next action is move forward 25 cm

Example 2:
User Instruction: "turn to face the plant on your right"
*[Image shows a plant to the right]*
Your Output: The next action is turn right 30 degree

Example 3:
User Instruction: "stop in front of the sofa"
*[Image shows the robot is right in front of the sofa]*
Your Output: The next action is stop

--- CURRENT TASK ---
User Instruction: "{instruction}"
*[Image shows the current camera view and history]*
Your Output:"""

            # version 2
#             navigation_template = """You are a navigation robot. Follow the user's instruction.
# Your response MUST be ONLY ONE of the formats shown in the examples below. Do not add explanations.

# --- EXAMPLES ---
# User Instruction: "go forward towards the door"
# Output: The next action is move forward 25 cm

# User Instruction: "turn to face the plant on your right"
# Output: The next action is turn right 30 degree

# User Instruction: "stop in front of the sofa"
# Output: The next action is stop

# --- CURRENT TASK ---
# User Instruction: "{instruction}"
# Output:"""
            # version3
#             navigation_template = """You are a navigation robot. Provide the next action command based on the user instruction.
# Your entire response must be ONLY one of the following four formats, with units in cm or degrees:
# - `The next action is move forward XX cm`
# - `The next action is turn left XX degree`
# - `The next action is turn right XX degree`
# - `The next action is stop`

# Instruction: "{instruction}"
# Output:"""

            text = navigation_template.format(instruction=user_instruction)

        # print(F"##### INPUT #####\n{text}\n###############")

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", 
                         "image": path if path.startswith("http") else f"file://{os.path.abspath(path)}"
                        } for path in image
                    ],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if enable_thinking:
            # print("Thinking enabled.")
            text = f"{text}<think>"
        else:
            # print("Thinking disabled.")
            text = f"{text}<think></think><answer>"

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # --- 2. 添加计时功能 ---
        print("Running inference ...")
        start_time = time.time() # 开始计时

        generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        end_time = time.time() # 结束计时
        inference_time = end_time - start_time
        print(f"Inference took {inference_time:.4f} seconds.")
        # --- 计时功能结束 ---

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if enable_thinking:
            parts = output_text[0].split("</think>")
            thinking_text = parts[0].replace("<think>", "").strip()
            answer_text = parts[1].replace("<answer>", "").replace("</answer>", "").strip() if len(parts) > 1 else ""
        else:
            thinking_text = ""
            answer_text = output_text[0].replace("<answer>", "").replace("</answer>", "").strip()

        return {
            "thinking": thinking_text,
            "answer": answer_text,
            "inference_time_seconds": inference_time # <<< 3. 将耗时加入返回结果
        }

if __name__ == "__main__":
    # --- 模型和图片路径配置 ---
    # model_path = "BAAI/RoboBrain2.0-7B" # 如果想从网络加载
    model_path = "/data/model_zoo/RoboBrain2.0-7B" 
    original_image_path = "img/image1.jpg" 
    
    # 临时文件夹存在
    os.makedirs("img/tmp", exist_ok=True)
    resized_image_path = "img/tmp/resized_image.jpg"

    # --- 实例化模型 ---
    model = SimpleInference(model_path)

    prompt = "Go to the white cabinet on the left"
    
    # 缩小图片以防止显存不足
    resized_path = model._resize_image(original_image_path, resized_image_path)

    # --- 执行推理 ---
    pred = model.inference(
        text=prompt, 
        image=resized_path, 
        task="navigation",
        enable_thinking = False 
    )
    print(pred)

    print("\n" + "="*20 + " RESULT " + "="*20)
    print(f"Final Answer: {pred['answer']}")
    print(f"Inference Time: {pred['inference_time_seconds']:.4f} seconds")
    print("="*48)