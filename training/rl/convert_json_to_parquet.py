import json
import os
from datasets import Dataset, Features, Value, Sequence, Image
import re
from typing import Union, Tuple, List

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_agent.tools.base import BaseTool, register_tool

from tqdm import tqdm

    
@register_tool("computer_use")
class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "scroll",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click"]:
            return self._mouse_click(action)
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str):
        raise NotImplementedError()

    def _key(self, keys: List[str]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _mouse_move(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _scroll(self, pixels: int):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
    
def extract_img_size(text: str):
    """
    Extracts the screen resolution from a string and returns it as [width, height].
    Example: "The screen's resolution is 1932x1092." -> [1932, 1092]
    """
    match = re.search(r"(\d+)\s*x\s*(\d+)", text)
    assert match is not None
    if match:
        w, h = match.groups()
        return [int(w), int(h)]
    return None

def json_to_parquet(json_path: str) -> str:
    """
    Convert a JSON array of dicts into a single .parquet file.
    The "images" field will be stored as Image() so it's loaded as PIL.Image.Image.
    Args:
        json_path (str): Path to the input JSON file.
    Returns:
        str: Path to the saved parquet file.
    """
    # 1) Load JSON array
    with open(json_path, "r") as f:
        data = json.load(f)
    new_data = []
    for d in tqdm(data):
        prompt = []
        image_size = extract_img_size(d['system'])
        
        computer_use = ComputerUse(
        cfg={"display_width_px": image_size[0], "display_height_px": image_size[1]})

        system_message = NousFnCallPrompt().preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
        ],
        functions=[computer_use.function],
        lang=None,
    )
        system_message = system_message[0].model_dump()
        
        
        correct_system_prompt = ""
        
        for item in system_message['content']:
            correct_system_prompt += item['text']
        
        if d['system'].strip() != correct_system_prompt.strip():
            d['system'] = correct_system_prompt
            # input()
            
        prompt.append({"content": d['system'],
                    "role": "system"})
        prompt.append({"content": "<image>" + d['instruction'],
                    "role": "user"})
        gt_bbox = d['gt_bbox']
        reward_model = {
            "ground_truth": {
                "x1": gt_bbox[0],
                "x2": gt_bbox[2],
                "y1": gt_bbox[1],
                "y2": gt_bbox[3]
            }
        }
        new_data.append({
            "data_source": "point",
            "prompt": json.dumps(prompt),
            "images": d['images'],
            "ability": "gui_grounding",
            "reward_model": reward_model,
            "extra_info": {"gt_response": d['gt_response'], "gt_bbox": d["gt_bbox"], "image_size": image_size}
        })
    features = Features({
        "data_source": Value("string"),
        "prompt": Value("string"),
        "images": Sequence(Image()),
        "ability": Value("string"),
        "reward_model": Features({
            "ground_truth": Features({
                "x1": Value("int64"),
                "x2": Value("int64"),
                "y1": Value("int64"),
                "y2": Value("int64"),
            })
        }),
        "extra_info": Features({
            "gt_response": Value("string"),
            "gt_bbox": Sequence(Value("int64")),
            "image_size": Sequence(Value("int64")),
        }),
    })
    # 3) Build dataset and cast schema
    ds = Dataset.from_list(new_data, features=features)
    # ds = Dataset.from_list(new_data).cast(features)
    # 4) Change extension to .parquet
    out_file = os.path.splitext(json_path)[0] + ".parquet"
    # 5) Save to parquet
    ds.to_parquet(out_file)
    print(f"Saved to: {out_file}")
    return out_file


paths = [
    'data/groundcua_train.json',
]

for path in paths:
    print(path)
    json_to_parquet(path)