import torch
from transformers import AutoProcessor

import json
import re
import os
from PIL import Image
import tempfile

from gui_actor.constants import chat_template
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference

from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

def extract_bbox(s):
    # First try to parse JSON to find "bbox_2d"
    try:
        # Look for a JSON code block in case the string is wrapped in triple backticks
        json_block = None
        m = re.search(r"```json(.*?)```", s, re.DOTALL)
        if m:
            json_block = m.group(1).strip()
        else:
            # If no explicit JSON block is found, assume the entire string might be JSON
            json_block = s.strip()

        data = json.loads(json_block)
        # If the data is a list, look for a dictionary with the "bbox_2d" key
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "bbox_2d" in item:
                    bbox = item["bbox_2d"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        return (bbox[0], bbox[1]), (bbox[2], bbox[3])
        elif isinstance(data, dict) and "bbox_2d" in data:
            bbox = data["bbox_2d"]
            if isinstance(bbox, list) and len(bbox) == 4:
                return (bbox[0], bbox[1]), (bbox[2], bbox[3])
    except Exception:
        # If JSON parsing fails, we'll fall back to regex extraction
        pass

    # Regex patterns to match bounding boxes in the given string format.
    pattern1 = r"<\|box_start\|\>\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]<\|box_end\|\>"
    pattern2 = r"<\|box_start\|\>\(\s*(\d+),\s*(\d+)\s*\),\(\s*(\d+),\s*(\d+)\s*\)<\|box_end\|\>"

    matches = re.findall(pattern1, s)
    if not matches:
        matches = re.findall(pattern2, s)

    if matches:
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))

    # If nothing was found, return None
    return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name

def extract_bbox_simple(s):
    # Regular expression to match bounding box in the format [x0, y0, x1, y1]
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    # Find all matches in the text
    matches = re.findall(pattern, s)
    
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    
    return None  # Return None if no bbox is found

class GUI_Actor():
    
    def __init__(self, model_name_or_path="microsoft/GUI-Actor-7B-Qwen2.5-VL", **kwargs):
        self.load_model(model_name_or_path)

    
    def load_model(self, model_name_or_path="microsoft/GUI-Actor-7B-Qwen2.5-VL"):
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()

    def set_response_type(self, **kwargs):
        pass

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, conversation, image, img_size=None):
        
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        else:
            assert isinstance(image, Image.Image)
        
        
        img_width, img_height = image.size
        
        for msg in conversation:
            if msg['role'] == 'user':
                msg['content'] = [{"type": "image", "image": image}] + [{"type": "text", "text": msg['content']}]
            else:
                msg['content'] = [{"type": "text", "text": msg['content']}]


        pred = inference(conversation, self.model, self.tokenizer, self.processor, use_placeholder=True, topk=3)
        click_point = pred["topk_points"][0]
        px, py = click_point

        print('### RESPONSE:')
        response = {"predicted_coords": [px * img_width, py * img_height]}
        print(f"Predicted response: {response}")
        
        # response = '<tool_call>\n{"name": "computer_use", "arguments": {"action": ' + response
        
        return json.dumps(response)
    
    def get_responses(self, args, messages_list, images):
        res = []
        self.debug_mode = args.debug_mode == 1
        for messages, image in tqdm(zip(messages_list, images), desc="Generating responses"):
        
            # Process each instance and generate a response
            response = self.ground_only_positive(conversation=messages,
                                                 image=image)
            res.append(response)
            
        return res

