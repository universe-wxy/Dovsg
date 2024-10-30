import sys
sys.path.append("/home/yanzj/workspace/code/DovSG")
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
from evaluation.eval_utils import scene_change_detection_system_prompt, scene_change_detection_user_prompt
from evaluation.eval_utils import scene_change_detection_user_prompt_1, scene_change_detection_user_prompt_2

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EvalSceneChange:
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

    def gpt4o_eval_scene_chagne_detection(self):
        view_dataset_path = self.memory_floder / "step_0" / "view_dataset.pkl"
        assert view_dataset_path.exists()

        with open(view_dataset_path, 'rb') as f:
            view_dataset = pickle.load(f)
        
        featurematch = RGBFeatureMatch()
        append_length = view_dataset.append_length_log[-1]
        images = view_dataset.images[-append_length:]
        lightglue_features = featurematch.extract_memory_features(images=images, features=None)

        obs_paths = self.find_npy_files(memory_floder)
        for idx, obs_path in enumerate(obs_paths):
            if idx == 0:
                continue
            observations = np.load(obs_path, allow_pickle=True).item()
            ref_image_paths = []
            new_image_paths = []
            path_parts = obs_path.parts
            out_path = self.output_floder / path_parts[7] / path_parts[9] / path_parts[10] / obs_path.stem
            new_img_out_path = out_path / "new"
            ref_img_out_path = out_path / "ref"
            new_img_out_path.mkdir(parents=True, exist_ok=True)
            ref_img_out_path.mkdir(parents=True, exist_ok=True)

            for i in range(len(observations["wrist"])):
                rgb = observations["wrist"][i]["rgb"]
                new_image = (rgb * 255).astype(np.uint8)
                best_index, _ = featurematch.find_most_similar_image(
                    new_image, features=lightglue_features)
                ref_image = images[best_index]
                
                ref_image_pil = Image.fromarray(ref_image)
                new_image_pil = Image.fromarray(new_image)

                ref_image_pil = ref_image_pil.resize(self.resize_resolution, Image.ANTIALIAS)
                new_image_pil = new_image_pil.resize(self.resize_resolution, Image.ANTIALIAS)

                ref_image_pil.save(ref_img_out_path / f"{i}.png")
                new_image_pil.save(new_img_out_path / f"{i}.png")
                
                ref_image_paths.append(ref_img_out_path / f"{i}.png")
                new_image_paths.append(new_img_out_path / f"{i}.png")
            
            # avoid gpt memory
            chat_messages = deepcopy(self.get_prompt(ref_image_paths=ref_image_paths, new_image_paths=new_image_paths)[:])

            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_messages,
                timeout=self.timeout
            )
            print(chat_completion.choices[0].message.content)

            response_path = out_path / "scene_chagne_detection.json"
            with open(response_path, "w") as f:
                json.dump(chat_completion.choices[0].message.content, f)
            # response = chat_completion.choices[0].message.content


    def get_prompt(self, ref_image_paths: List[Path], new_image_paths: List[Path]):

        prompt_json = [
            {
                "role": "system",
                "content": scene_change_detection_system_prompt
            },
            {
                "role": "user",
                "content": scene_change_detection_user_prompt
            }
        ]

        
        prompt_ref = {"role": "user", "content": [{"type": "text", "text": scene_change_detection_user_prompt_1}]}
        prompt_new = {"role": "user", "content": [{"type": "text", "text": scene_change_detection_user_prompt_2}]}
        ref_query_images = [encode_image(img_path) for img_path in ref_image_paths]
        new_query_images = [encode_image(img_path) for img_path in new_image_paths]
        for ref_query_image in ref_query_images:
            prompt_ref["content"].append({
                "type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{ref_query_image}"
                }
            })
        
        for new_query_image in new_query_images:
            prompt_new["content"].append({
                "type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{new_query_image}"
                }
            })
        
        prompt_json.append(prompt_ref)
        prompt_json.append(prompt_new)

        return prompt_json

if __name__ == "__main__":
    memory_floder = Path("/home/yanzj/workspace/code/DovSG/data/company_room_1_10_5_new/memory/3_0.1_0.01_True_0.2_0.5")
    output_floder = Path("evaluation/output")
    eval_scene_change = EvalSceneChange(memory_floder=memory_floder, output_floder=output_floder)
    eval_scene_change.gpt4o_eval_scene_chagne_detection()