from gguf import Union
import torch
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from tqdm import tqdm

import json
import re
import os
from PIL import Image
import tempfile


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


SYSTEM_PROMPT = """You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {img_width}x{img_height}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `wait`: Wait specified seconds for the change to happen.", "enum": ["mouse_move", "wait"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}}}, "required": ["action"], "type": "object"}}}}}}
{{"type": "function", "function": {{"name": "zoom_in", "description": "Zoom into the current image by cropping a selected quadrant from a 2×2 grid and upsampling it. The crop is specified by pos = [row, col] with 0-based indices (row: 0=top, 1=bottom; col: 0=left, 1=right). Returns the zoomed crop as an <image> observation for the next turn.", "parameters": {{"properties": {{"pos": {{"type": "array", "items": [{{"type": "integer", "enum": [0, 1], "description": "row index of the quadrant (0=top, 1=bottom)"}}, {{"type": "integer", "enum": [0, 1], "description": "column index of the quadrant (0=left, 1=right)"}}], "minItems": 2, "maxItems": 2, "description": "The [row, col] indices of the quadrant to zoom into."}}}}, "type": "object", "required": ["pos"]}}}}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""



class ZoomInQwen25VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", max_length=2048, temperature=0.0, max_image_tokens=0.0, cap_pixels=0.0):
    
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if hasattr(config, "text_config"):        
            if isinstance(config.text_config, dict):
                config.text_config = AutoConfig.for_model("qwen2_5_vl").from_dict(config.text_config)

        if hasattr(config, "vision_config"):    
            if isinstance(config.vision_config, dict):
                config.vision_config = AutoConfig.for_model("qwen2_5_vl").from_dict(config.vision_config)


        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=config,            
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_name_or_path,        
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        #     trust_remote_code=True
        # ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        

        self.model.generation_config.temperature = temperature
        self.model.generation_config.do_sample = False if temperature == 0.0 else True
        self.model.generation_config.max_new_tokens = max_length
        self.model.generation_config.use_cache = True
        
        self.max_image_tokens = max_image_tokens
        self.cap_pixels = cap_pixels
        self.max_num_responses = 1
        
        assert not (self.max_image_tokens > 0 and self.cap_pixels > 0), "Only one of max_image_tokens and cap_pixels can be set."
        
        if self.max_image_tokens > 0:
            self.max_pixels = int(28 * 28 * self.max_image_tokens)
        elif cap_pixels > 0:
            self.max_pixels = int(cap_pixels * 1_000_000)
        else:
            self.max_pixels = int(12 * 1_000_000)
    
    
    def zoom_in(self, img, rc):
        """
        Crop a quadrant from `img` based on (row, col) where row, col ∈ {0,1},
        then resize the crop back to the original image size.

        Args:
            img: PIL Image (RGB/RGBA/L mode all fine).
            rc: (row, col) where row=0 is top half, row=1 bottom half;
                col=0 is left half, col=1 right half.

        Returns:
            resized_crop: the cropped quadrant resized to original (W, H).
            crop_box: the box used for cropping in (left, upper, right, lower).
        """
        row, col = rc
        row, col = int(row), int(col)
        if row not in (0, 1) or col not in (0, 1):
            raise ValueError("row and col must be 0 or 1")

        W, H = img.size

        # Split widths/heights in a way that handles odd sizes deterministically.
        # Left/top get the floor half; right/bottom get the remaining pixels.
        w0 = W // 2
        w1 = W - w0
        h0 = H // 2
        h1 = H - h0

        # Compute crop box based on (row, col)
        # row=0 -> y from 0 to h0, row=1 -> y from h0 to H
        # col=0 -> x from 0 to w0, col=1 -> x from w0 to W
        y0 = 0 if row == 0 else h0
        y1 = h0 if row == 0 else H
        x0 = 0 if col == 0 else w0
        x1 = w0 if col == 0 else W

        crop_box = (x0, y0, x1, y1)
        cropped = img.crop(crop_box)

        # Resize back to original size
        # Use a good resampler for upscaling (or downscaling if odd split)
        # resized_crop = cropped.resize((W, H), Image.BICUBIC)

        return cropped


    def generate_until(self, instances):
        res = []
        for instance in tqdm(instances):
            # Process each instance and generate a response
            response = self.ground_only_positive(instance)
            res.append(response)
        return res

    def ground_only_positive(self, request):
        
            # BUILD the same “Chat” payload you had before:
        
        instruction, tools, image_path, resized_args = (
                request["instruction"],
                request["tools"],
                request["image_path"],
                request["resized_args"],
            )
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        # image_inputs, video_inputs = process_vision_info(messages)
        image = Image.open(image_path).convert('RGB')
        
        # assert image.size[0] == resized_args[0]
        # assert image.size[1] == resized_args[1]

        org_image = image.copy()
            
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            min_pixels=56 * 56,
            max_pixels=self.max_pixels,
            # max_pixels=2_100_00,
        )
        print("before:", image.size[0] * image.size[1])
        image = image.resize((resized_width, resized_height))
        print("after:", image.size[0] * image.size[1])
        
        img_width, img_height = resized_width, resized_height

        # img_width, img_height = img_size
        
        # prompt_origin = 'Output the bounding box in the image of the UI element corresponding to the instruction "{}" with grounding. The coordinates should be relative ranging from 0 to 1000, relative to the actually image length and width (i.e. all values (x and y) in a range [0, 1000]).'
        # full_prompt = prompt_origin.format(instruction)
        # full_prompt = f'Click the UI element corresponding to the instruction "{instruction}". The screenshot given is {img_width}x{img_height}.'
        full_prompt = f'Please complete the following tasks via mouse move or wait: "{instruction}".'
        # full_prompt = f'{instruction}'
        
        messages = [
            {
            "role": "system",
            "content": SYSTEM_PROMPT.format(img_width=img_width, img_height=img_height)
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # "image": image_path,
                        "image": image,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        
        input_images = [image]
        
        responses = []
        zoom_ins = []
        num_responses = 0
        click_point = None
        zoom_in_pos = None
        got_terminal_action = False
        
        while True:
            
            if num_responses > self.max_num_responses:
                break
            
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_input],
                images=input_images,
                # images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            responses.append(response)
            
            if '<tool_call>' in response and '</tool_call>' in response:
                response = response.replace('\n', '').strip()
            
            if "computer_use" in response:
                got_terminal_action = True
                
                # '<tool_call>\n{"name": "computer_use", "arguments": {"action": "mouse_move", "coordinate": [596, 347]}}\n</tool_call>'
                try:
                    
                    action = json.loads(response.split('<tool_call>')[1].split('</tool_call>')[0])
                    click_point = action.get('arguments', {"coordinate": [-1, -1]})['coordinate']
                except:
                    print('******* computer_use json string had a problem *******')
                    match = re.search(r'"coordinate":\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', response)
                    if match:
                        click_point = [int(match.group(1)), int(match.group(2))]
                    else:
                        click_point = [-1, -1]
                break

            
            elif "zoom_in" in response:
                # '<tool_call>\n{"name": "zoom_in", "arguments": {"pos": [0, 0]}}\n</tool_call>'
                try:
                    action = json.loads(response.split('<tool_call>')[1].split('</tool_call>')[0])
                    zoom_in_pos = action.get('arguments', {"pos": [-1, -1]})['pos']
                except:
                    print('******* zoom_in json string had a problem *******')
                    match = re.search(r'"pos":\s*\[\s*([01])\s*,\s*([01])\s*\]', response)
                    if match:
                        zoom_in_pos = [int(match.group(1)), int(match.group(2))]
                    else:
                        zoom_in_pos = [-1, -1]

                if zoom_in_pos == [-1, -1]:
                    break
                else:
                    try:
                        zoomed_in_img = self.zoom_in(org_image, zoom_in_pos).resize(image.size)
                    except:
                        break
                    zoom_ins.append(zoom_in_pos)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user",
                        "content": [{
                            "type": "image",
                            "image": zoomed_in_img,
                            },
                            {"type": "text", "text": "<image>"}]}
                        )
                
                    input_images.append(zoomed_in_img)
            else:
                break
                
            num_responses += 1

        
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": '\n'.join(responses),
            "bbox": None,
            "point": None
        }
        
        if not got_terminal_action or \
            ((zoom_in_pos is None or zoom_in_pos == [-1, -1]) and \
            (click_point is None or click_point == [-1, -1])):
                return result_dict

        assert len(zoom_ins) == len(responses) - 1

        
        if click_point is None:
            print('------- no available tool_call for computer_use :( --------')
            click_point = [-1, -1]

        if len(click_point) == 2:
            if len(zoom_ins) != 0:    
                click_point[0] = 0.5 * (click_point[0] / img_width) + (0.5 * zoom_in_pos[1])
                click_point[1] = 0.5 * (click_point[1] / img_height) + (0.5 * zoom_in_pos[0])
            else:
                click_point[0] = click_point[0] / img_width
                click_point[1] = click_point[1] / img_height
                
            result_dict["point"] = click_point
        
        click_point_resized = [click_point[0] * resized_args[0], click_point[1] * resized_args[1]]
        
        # return result_dict['raw_response']
        return {"predicted_coords": click_point_resized,
                "raw_respons": result_dict['raw_response']}
    

    def chat(self, messages, images, **kwargs):
        pass
     
    
    def get_responses(self, args, messages_list, images):
        res = []
        self.debug_mode = args.debug_mode == 1
        for messages, image in tqdm(zip(messages_list, images), desc="Generating responses"):
            # Process each instance and generate a response
            response = self.chat(messages, [image])
            res.append(response)
        return res