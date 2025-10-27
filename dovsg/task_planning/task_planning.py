import time
import json
import argparse
import re
from pathlib import Path
from typing import Union
import copy
from openai import OpenAI
# from prompts_1 import system_prompt
# from prompts_1 import (
#     description_1,
#     description_2,
#     description_3,
#     description_4,
#     description_5,
# )
# from prompts_1 import response_1, response_2, response_3, response_4, response_5

from prompts_2 import system_prompt
from prompts_2 import (
    description_1,
    description_2,
    description_3,
    description_4,
    description_5,
)
from prompts_2 import response_1, response_2, response_3, response_4, response_5


def safe_filename(text: str) -> str:
    text = text.strip().lstrip("+")
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._-]+", "_", text)
    return text[:80] or "task_result"


class TaskPlanning:
    def __init__(
        self,
        timeout: int = 25,  # Timeout in seconds
        save_dir: Union[Path, None] = None,
    ):
        self.timeout = timeout
        self.save_dir = save_dir
        self.client = OpenAI(
            api_key="sk-63gz3Qle25G87FtCtDYMzwqvfa1CIBA72PxS5ACg5AbHkJbI",
            base_url="https://api.agicto.cn/v1",
        )

    def get_response(self, description: str, retry_num: int = 3):

        chat_messages = copy.deepcopy(self.get_prompt())

        chat_messages.append({"role": "user", "content": description})

        response = None
        retry = 1
        while retry <= retry_num:
            try:
                chat_completion = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=chat_messages, timeout=self.timeout # type: ignore
                )
                response_text = chat_completion.choices[0].message.content
                if "subtasks" in response_text and "Reasoning" in response_text: # type: ignore
                    response = json.loads(response_text) # type: ignore
                    break
                else:
                    print("response format error, retrying...")
                    retry += 1
            except Exception as e:
                print(f"Attempt {retry} failed: {str(e)}")
                retry += 1
                time.sleep(1)

        assert response is not None, "Can't get valid response from OpenAI."

        filename = safe_filename(description) + ".json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)
        print(f"Subtasks have be saved: {filename}")

        return response

    def get_prompt(self):
        prompt_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description_1},
            {"role": "assistant", "content": response_1},
            {"role": "user", "content": description_2},
            {"role": "assistant", "content": response_2},
            {"role": "user", "content": description_3},
            {"role": "assistant", "content": response_3},
            {"role": "user", "content": description_4},
            {"role": "assistant", "content": response_4},
            {"role": "user", "content": description_5},
            {"role": "assistant", "content": response_5},
        ]
        return prompt_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute task planning based on description."
    )
    parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="Please move the red pepper to the plate, then move the green pepper to plate",
    )
    args = parser.parse_args()

    task_desc = args.task_description.strip()
    print(f"Task Description: {task_desc}\n")

    planner = TaskPlanning()
    result = planner.get_response(description=task_desc)
