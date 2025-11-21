"""
GroundCUA Helper Module
Contains constants and utility functions for GroundNext models.
"""

from .version import __version__

GROUNDNEXT_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {width}x{height}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button.\n* `middle_click`: Click the middle mouse button.\n* `double_click`: Double-click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click_drag`, `action=left_click`, `action=right_click`, `action=double_click`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

# Default generation parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_NEW_TOKENS = 64
MIN_PIXELS = 78_400
MAX_PIXELS = 6_000_000
SAMPLE_URL = "https://huggingface.co/datasets/ServiceNow/GroundCUA/resolve/main/images/7-Zip/001f0079a489909eb94e47c2374b7bf36ab1842e314592ce30a34d18a54eb1df.png"

def set_generation_config(model, do_sample=True, use_cache=False, temperature=DEFAULT_TEMPERATURE):
    model.generation_config.do_sample = do_sample
    model.generation_config.use_cache = use_cache
    model.generation_config.temperature = temperature
    
def prepare_image(image, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """
    Resize image using smart_resize for optimal model performance.
    
    Args:
        image: PIL Image object
        min_pixels: Minimum number of pixels (default: 78,400)
        max_pixels: Maximum number of pixels (default: 6,000,000)
    
    Returns:
        Resized PIL Image and (width, height) tuple
    """
    from qwen_vl_utils.vision_process import smart_resize
    
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    resized_image = image.resize((resized_width, resized_height))
    return resized_image, (resized_width, resized_height)


def create_messages(instruction, image, width, height):
    """
    Create message format for GroundNext models.
    
    Args:
        instruction: Text instruction for grounding task
        image: PIL Image object
        width: Image width
        height: Image height
    
    Returns:
        List of messages in the correct format
    """
    return [
        {
            "role": "system",
            "content": GROUNDNEXT_SYSTEM_PROMPT.format(width=width, height=height)
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

def trim_ids(ids, input_ids):
    return [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, ids)]


__all__ = [
    "__version__",
    "GROUNDNEXT_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_NEW_TOKENS",
    "MIN_PIXELS",
    "MAX_PIXELS",
    "SAMPLE_URL",
    "prepare_image",
    "create_messages",
    "set_generation_config",
]

