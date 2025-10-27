import time
import os
import json
from pathlib import Path
from typing import Union
import copy
from openai import OpenAI
from dovsg.memory.scene_graph.scene_graph_processer import SceneGraph
from dovsg.task_planning.prompts import system_prompt
from dovsg.task_planning.prompts import description_1, description_2, description_3, description_4, description_5
from dovsg.task_planning.prompts import response_1, response_2, response_3, response_4, response_5

class TaskPlanning:
    def __init__(
        self,
        timeout: int=25,  # Timeout in seconds,
        save_dir: Union[Path, None] = None
    ):
        self.timeout = timeout
        self.save_dir = save_dir
        self.client = OpenAI(
            api_key="sk-63gz3Qle25G87FtCtDYMzwqvfa1CIBA72PxS5ACg5AbHkJbI",
            base_url="https://api.agicto.cn/v1"
        )

    def get_response(
        self, 
        description: str,
        # instance_scene_graph: SceneGraph,
        save_response: bool=True,
        retry_num: int=3
    ):
        if save_response:
            self.save_dir.mkdir(exist_ok=True)
        
        response = None
        response_save_path = self.save_dir / f"{description}.json"
        if response_save_path.exists():
            with open(response_save_path, "r") as f:
                response = json.load(f)["response"]
        else:
            # avoid gpt memory
            chat_messages = copy.deepcopy(self.get_prompt()[:])

            chat_messages.append(
                {
                    "role": "user", 
                    "content": description
                }
            )
            retry = 1
            while retry < retry_num:
                try:
                    chat_completion = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=chat_messages,
                        timeout=self.timeout
                    )
                    response = chat_completion.choices[0].message.content
                    if 'subtasks' in response and 'Reasoning' in response:
                        response = json.loads(chat_completion.choices[0].message.content)
                        with open(response_save_path, "w") as f:
                            json.dump({
                                "input": chat_messages,
                                "response": response
                            }, f, indent=4)
                        break
                    else:
                        print("response format error.")
                        retry += 1
                except Exception as e:
                    print(f"When get response from gpt, an unexpected error occurred: {str(e)}")
        

            assert retry < retry_num, "can't get reponse from openai."

        return response
        
    
    def get_prompt(self):
        prompt_json = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": description_1
            },
            {
                "role": "assistant",
                "content": response_1
            },
            {
                "role": "user",
                "content": description_2
            },
            {
                "role": "assistant",
                "content": response_2
            },
            {
                "role": "user",
                "content": description_3
            },
            {
                "role": "assistant",
                "content": response_3
            },
            {
                "role": "user",
                "content": description_4
            },
            {
                "role": "assistant",
                "content": response_4
            },
            {
                "role": "user",
                "content": description_5
            },
            {
                "role": "assistant",
                "content": response_5
            }
        ]
        return prompt_json