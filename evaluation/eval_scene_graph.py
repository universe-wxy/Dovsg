import sys
sys.path.append("/home/yanzj/workspace/code1/DovSG")
from dovsg.scripts.rgb_feature_match import RGBFeatureMatch
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Union, List
from openai import OpenAI
from PIL import Image
import json
from copy import deepcopy
from evaluation.eval_utils import encode_image
from evaluation.eval_utils import scene_graph_generation_system_prompt
from evaluation.eval_utils import scene_graph_generation_user_prompt
from evaluation.eval_utils import scene_graph_generation_user_prompt_1

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EvalSceneGraph:
    def __init__(
        self,
        memory_floder: str, 
        output_floder: str,
        resize_resolution: tuple = (640, 300),
        timeout: int=60,  # Timeout in seconds
    ):
        self.memory_floder = memory_floder
        self.output_floder = output_floder
        self.resize_resolution = resize_resolution
        self.timeout = timeout
    
    def find_npy_files(self, directory):
        npy_files = []
        for root, dirs, files in os.walk(str(directory)):
            for file in files:
                if file.endswith(".npy"):
                    npy_files.append(Path(root) / file)
        return npy_files

    def gpt4o_eval_scene_graph_generation(self):
        obs_paths = self.find_npy_files(memory_floder)
        for idx, obs_path in enumerate(obs_paths):
            if idx == 0:
                continue
            observations = np.load(obs_path, allow_pickle=True).item()
            new_image_paths = []
            path_parts = obs_path.parts
            out_path = self.output_floder / path_parts[7] / path_parts[9] / path_parts[10] / obs_path.stem
            new_img_out_path = out_path / "new"
            new_img_out_path.mkdir(parents=True, exist_ok=True)

            for i in range(len(observations["wrist"])):
                rgb = observations["wrist"][i]["rgb"]
                new_image = (rgb * 255).astype(np.uint8)

                new_image_pil = Image.fromarray(new_image)
                new_image_pil = new_image_pil.resize(self.resize_resolution, Image.ANTIALIAS)
                new_image_pil.save(new_img_out_path / f"{i}.png")
                
                new_image_paths.append(new_img_out_path / f"{i}.png")
            
            # avoid gpt memory
            chat_messages = deepcopy(self.get_prompt(new_image_paths=new_image_paths)[:])

            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_messages,
                timeout=self.timeout
            )
            print(chat_completion.choices[0].message.content)

            response_path = out_path / "scene_graph_response.json"
            with open(response_path, "w") as f:
                json.dump(chat_completion.choices[0].message.content, f)


    def get_prompt(self, new_image_paths: List[Path]):

        prompt_json = [
            {
                "role": "system",
                "content": scene_graph_generation_system_prompt
            },
            {
                "role": "user",
                "content": scene_graph_generation_user_prompt
            }
        ]

        prompt_new = {"role": "user", "content": [{"type": "text", "text": scene_graph_generation_user_prompt_1}]}
        new_query_images = [encode_image(img_path) for img_path in new_image_paths]
        
        for new_query_image in new_query_images:
            prompt_new["content"].append({
                "type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{new_query_image}"
                }
            })
        
        prompt_json.append(prompt_new)

        return prompt_json

if __name__ == "__main__":
    memory_floder = Path("/home/yanzj/workspace/code1/DovSG/data/company_room_1_10_5_new/memory/3_0.1_0.01_True_0.2_0.5")
    output_floder = Path("evaluation/output")
    eval_scene_change = EvalSceneGraph(memory_floder=memory_floder, output_floder=output_floder)
    eval_scene_change.gpt4o_eval_scene_graph_generation()