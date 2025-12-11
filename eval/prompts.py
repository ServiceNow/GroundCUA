import json
from typing import Dict, List, Any, Tuple, Optional
import re


def extract_and_parse_json(input_string: str, wrapper: str) -> Optional[List]:
    """
    Attempt to extract and parse a JSON array from a string using a given pair of wrapper characters.
    
    The function searches for the first occurrence of the start wrapper and the last occurrence
    of the end wrapper, and tries to parse the substring between them as JSON.
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)
    end_index = input_string.rfind(end_char)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None

    json_string = input_string[start_index:end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """
    Ray casting algorithm to determine if a point lies inside a polygon.
    
    Args:
        point: Point coordinates as [x, y].
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...].
    
    Returns:
        True if the point is inside the polygon; False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


# def is_point_in_polygon(point, polygon):
#     x, y = point
#     n = len(polygon) // 2
#     inside = False

#     j = n - 1
#     for i in range(n):
#         xi, yi = polygon[i * 2], polygon[i * 2 + 1]
#         xj, yj = polygon[j * 2], polygon[j * 2 + 1]

#         if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
#             inside = not inside
#         j = i

#     return inside



def is_point_inside_element(element: Dict, point: List[float]) -> bool:
    """
    Check whether a predicted point lies inside a ground-truth region.
    
    Args:
        element: Dictionary describing the region, may contain a bbox or a polygon.
        point: Predicted point coordinates [x, y].
    
    Returns:
        True if the point is inside the element area; False otherwise.
    """
    # Axis-aligned bounding box
    if "x1" in element and "y1" in element and "x2" in element and "y2" in element:
        return (element["x1"] <= point[0] <= element["x2"] and 
                element["y1"] <= point[1] <= element["y2"])
    
    # Polygon
    elif "polygon" in element:
        polygon = element["polygon"]
        if len(polygon) < 3:  # A polygon requires at least 3 vertices
            return False
        return point_in_polygon(point, polygon)

    elif "refusal" in element:
        return all(point[i] < 0 for i in range(2))
    
    # Unsupported format
    else:
        return False


class BasicPrompt:
    
    @staticmethod
    def get_metric_keys() -> Dict[str, str]:
        """
        Return the metric keys supported by this processor and their types.
        
        Returns:
            Mapping from metric key to type, where type is one of:
            - 'sum': metrics to be summed
            - 'avg': metrics to be averaged (sum and count are handled separately)
            - 'count': count metrics
        """
        return {
            "total": "sum",
            "correct": "sum", 
            "has_correct": "sum",
            "num_answers": "avg",  # This is averaged across samples
        }
    
    @staticmethod
    def get_accuracy_pairs() -> List[Tuple[str, str]]:
        """
        Return numerator/denominator pairs for accuracy calculations.
        
        Returns:
            List of (numerator, denominator) pairs.
        """
        return [
            ("correct", "total"),       # correct_accuracy = correct / total
            ("has_correct", "total"),   # has_correct_accuracy = has_correct / total
        ]
  
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = ('You are a helpful assistant.\n\n'
                        '# Tools\n\n'
                        'You may call one or more functions to assist with the user query.\n\n'
                        'You are provided with function signatures within <tools></tools> XML tags:\n'
                        '<tools>\n' 
                        '{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n'
                        '* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n'
                        '* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\'t open, try wait and taking another screenshot.\n'
                        '* The screen\'s resolution is {display_width_px}x{display_height_px}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n'
                        '* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n'
                        '* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n'
                        '* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n'
                        '* `type`: Type a string of text on the keyboard.\n'
                        '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `left_click`: Click the left mouse button.\n'
                        '* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `right_click`: Click the right mouse button.\n'
                        '* `middle_click`: Click the middle mouse button.\n'
                        '* `double_click`: Double-click the left mouse button.\n'
                        '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
                        '* `wait`: Wait specified seconds for the change to happen.\n'
                        '* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (distance from the left edge) and y (distance from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}}\n'
                        '</tools>\n\n'
                        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                        '<tool_call>\n'
                        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                        '</tool_call>\n')


            
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>Hover over the UI element '{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        
        # result = extract_and_parse_json(response, "[]")
        json_str = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
        
        try:
            json_response = json.loads(json_str)
            click_point = json_response.get("arguments").get('coordinate')
            assert len(click_point) == 2
            click_point = [int(x) for x in click_point]
        except (json.JSONDecodeError, KeyError) as e:
            click_point = re.findall(r"\d+", json_str)
            
            if len(click_point) == 2:
                click_point = [int(x) for x in click_point]
            else:
                click_point = None


        return click_point
        
    @staticmethod
    def calculate_metrics(sample: Dict, point: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single sample.
        
        Args:
            sample: Sample dictionary.
            predictions: List of predictions, format [{"point_2d": [x, y]}, ...].
            gt_bbox: Ground-truth bounding box, format {"x1": x1, "y1": y1, "x2": x2, "y2": y2} or {"polygon": [[x,y], ...]}.
        
        Returns:
            Dictionary containing metric values.
        """
        # Initialize metric dictionary dynamically based on metric_keys
        metric_keys = BasicPrompt.get_metric_keys()
        metrics = {}
        
        # Initialize base metrics
        for key, key_type in metric_keys.items():
            if key == "total":
                metrics[key] = 1  # Each sample counts as 1
            elif key_type in ["sum", "count"]:
                metrics[key] = 0
            elif key_type == "avg":
                metrics[key] = None
        
        # Add fixed field
        metrics["predictions"] = None
        
        # Handle samples without ground-truth (empty bbox)
        if not gt_bbox:
            if point is None:
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            else:
                metrics["correct"] = 0
                metrics["has_correct"] = 0
            # Accuracies are computed later; no need to set here
            return metrics
        
        # No predictions
        if point is None:
            return metrics
        
        try:
            # Check if any predicted point is correct
            
            if is_point_inside_element(gt_bbox, point):
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            
            metrics["predictions"] = point
            
        except (KeyError, IndexError, TypeError) as e:
            # Parsing error; return defaults in metrics
            pass
        
        # Accuracies are computed later; no need to set here
        return metrics
  

class InfiguiR1Prompt(BasicPrompt):
    """
    Default prompt processor for infigui-r1.
    """
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        if think_mode:
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags."
        else:
            system_prompt = "You are a helpful assistant."

        grounding_prompt = f'''The screen\'s resolution is {image_width}x{image_height}.
Point to the UI element most relevant to "{sample['instruction']}", output its coordinates using JSON format:\n```json\n[\n    {{"point_2d": [x, y], "label": "object name/description"}}\n]```'''

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f'''<image>{grounding_prompt}'''
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        """
        Extract [x, y] coordinates from a model response containing JSON with "point_2d".
        Returns None if no valid coordinates are found.
        """
        try:
            # Try to parse JSON directly
            json_response = json.loads(response)
            if isinstance(json_response, list) and len(json_response) > 0:
                coords = json_response[0].get("point_2d")
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    return [int(float(x)) for x in coords]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numbers with regex
        numbers = re.findall(r"-?\d+", response.split("point_2d")[-1])
        if len(numbers) >= 2:
            return [int(numbers[0]), int(numbers[1])]

        return None

class InfiguiG1Prompt(BasicPrompt):
    """
    Default prompt processor for infigui-g1.
    """
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        if think_mode:
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags."
        else:
            system_prompt = "You are a helpful assistant."
            
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f'''<image>The screen's resolution is {image_width}x{image_height}.
Locate the UI element(s) for "{sample['instruction']}", output the coordinates using JSON format: [{{"point_2d": [x, y]}}, ...]'''
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        if think_mode and "</think>" in response:
            response = response.split("</think>")[-1]
        
        result = extract_and_parse_json(response, "[]")
        return result
    
    @staticmethod
    def calculate_metrics(sample: Dict, predictions: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single sample.
        
        Args:
            sample: Sample dictionary.
            predictions: List of predictions, format [{"point_2d": [x, y]}, ...].
            gt_bbox: Ground-truth bounding box, format {"x1": x1, "y1": y1, "x2": x2, "y2": y2} or {"polygon": [[x,y], ...]}.
        
        Returns:
            Dictionary containing metric values.
        """
        # Initialize metric dictionary dynamically based on metric_keys
        metric_keys = InfiguiG1Prompt.get_metric_keys()
        metrics = {}
        
        # Initialize base metrics
        for key, key_type in metric_keys.items():
            if key == "total":
                metrics[key] = 1  # Each sample counts as 1
            elif key_type in ["sum", "count"]:
                metrics[key] = 0
            elif key_type == "avg":
                metrics[key] = None
        
        # Add fixed field
        metrics["predictions"] = None
        
        # Handle samples without ground-truth (empty bbox)
        if not gt_bbox:
            if predictions is None or len(predictions) == 0:
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            else:
                metrics["correct"] = 0
                metrics["has_correct"] = 0
            # Accuracies are computed later; no need to set here
            return metrics
        
        # No predictions
        if predictions is None or len(predictions) == 0:
            return metrics
        
        try:
            # Check if any predicted point is correct
            has_correct = 0
            for pred in predictions:
                point = pred["point_2d"]
                if is_point_inside_element(gt_bbox, point):
                    has_correct = 1
                    break
            
            metrics["has_correct"] = has_correct
            
            # Check the first prediction
            first_pred = predictions[0]["point_2d"]
            if is_point_inside_element(gt_bbox, first_pred):
                metrics["correct"] = 1
            
            metrics["predictions"] = first_pred
            metrics["num_answers"] = sum(1 for pred in predictions if "point_2d" in pred)
            
        except (KeyError, IndexError, TypeError) as e:
            # Parsing error; return defaults in metrics
            pass
        
        # Accuracies are computed later; no need to set here
        return metrics

class GroundCUAPrompt(BasicPrompt):
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = ('You are a helpful assistant.\n\n'
                        '# Tools\n\n'
                        'You may call one or more functions to assist with the user query.\n\n'
                        'You are provided with function signatures within <tools></tools> XML tags:\n'
                        '<tools>\n'
                        '{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n'
                        '* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n'
                        '* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\'t open, try wait and taking another screenshot.\n'
                        '* The screen\'s resolution is {display_width_px}x{display_height_px}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n'
                        '* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n'
                        '* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n'
                        '* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n'
                        '* `type`: Type a string of text on the keyboard.\n'
                        '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `left_click`: Click the left mouse button.\n'
                        '* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                        '* `right_click`: Click the right mouse button.\n'
                        '* `middle_click`: Click the middle mouse button.\n'
                        '* `double_click`: Double-click the left mouse button.\n'
                        '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
                        '* `wait`: Wait specified seconds for the change to happen.\n'
                        '* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (distance from the left edge) and y (distance from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}}\n'
                        '</tools>\n\n'
                        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                        '<tool_call>\n'
                        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                        '</tool_call>\n')

        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>{sample['instruction']}"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        
        # result = extract_and_parse_json(response, "[]")
        json_str = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
        
        try:
            json_response = json.loads(json_str)
            click_point = json_response.get("arguments").get('coordinate')
            assert len(click_point) == 2
            click_point = [int(x) for x in click_point]
        except (json.JSONDecodeError, KeyError) as e:
            click_point = re.findall(r"\d+", json_str)
            
            if len(click_point) == 2:
                click_point = [int(x) for x in click_point]
            else:
                click_point = None


        return click_point
   
class GTAPrompt(BasicPrompt):
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {display_height_px} and width {display_width_px}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''
            
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>'{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
            click_point = [tuple(map(int, match)) for match in matches][0]
        except:
            click_point = None
            
        return click_point

class OpenCUAPrompt(BasicPrompt):
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        system_prompt = (
            "You are a GUI agent. You are given a task and a screenshot of the screen. "
            "You need to perform a series of pyautogui actions to complete the task."
        )
            
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(display_width_px=image_width, display_height_px=image_height)
            },
            {
                "role": "user",
                "content": f"<image>'{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
            click_point = [tuple(map(int, match)) for match in matches][0]
        except:
            click_point = None
            
        return click_point


class GUIG2Prompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_prompt = "Outline the position corresponding to the instruction: {problem}. The output should be only [x1,y1,x2,y2]."
        
        messages = [
            {
                "role": "user",
                "content": f"<image>'{system_prompt.format(problem=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        bbox = None
        m = re.search(r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]", response)
        if m:
            bbox = [int(round(float(x))) for x in m.groups()]

        # 3) Last resort: grab any four numbers in order
        nums = re.findall(r"-?\d*\.?\d+", response)
        if len(nums) >= 4:
            bbox = [int(round(float(x))) for x in nums[:4]]

        if bbox is not None:
            click_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        else:
            click_point = None
        return click_point
        
class UGroundV1Prompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_prompt = """
  Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {description}

  Answer:"""
        
        messages = [
            {
                "role": "user",
                "content": f"<image>'{system_prompt.format(description=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
            click_point = [tuple(map(int, match)) for match in matches][0]
        except:
            click_point = None
            
        return click_point
        
class SEGUIPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        system_prompt =  ('You are a helpful assistant.\n\n'
                        '# Tools\n\n'
                        'You may call one or more functions to assist with the user query.\n\n'
                        'You are provided with function signatures within <tools></tools> XML tags:\n'
                        '<tools>\n'
                        '{{"type": "function", "function": {{"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.'
                        '* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu . You must click on desktop icons to start applications. ' 
                        '* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.'
                        '* The screen\'s resolution is {image_width}x{image_height}.'
                        '* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.}}}}'
                        '</tools>\n\n'
                        'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                        '<tool_call>\n'
                        '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                        '</tool_call>\n')
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(image_width=image_width, image_height=image_height)
            },
            {
                "role": "user",
                "content": f"<image>'{sample['instruction']}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        
        # result = extract_and_parse_json(response, "[]")
        json_str = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
        
        try:
            json_response = json.loads(json_str)
            click_point = json_response.get("arguments").get('coordinate')
            assert len(click_point) == 2
            click_point = [int(x) for x in click_point]
        except (json.JSONDecodeError, KeyError) as e:
            click_point = re.findall(r"\d+", json_str)
            
            if len(click_point) == 2:
                click_point = [int(x) for x in click_point]
            else:
                click_point = None


        return click_point
        
class GUIG1Prompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        user_prompt =  'Grounding instruction is:{question}. Help to locate and output its bbox coordinates using JSON format::\n```json\n[\n{{"point_2d": [x, y], "label": "object name/description"}}\n]```'

        messages = [
            {
                "role": "user",
                "content": f"<image>'{user_prompt.format(question=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        """
        Extract [x, y] coordinates from a model response containing JSON with "point_2d".
        Returns None if no valid coordinates are found.
        """
        try:
            # Try to parse JSON directly
            json_response = json.loads(response)
            if isinstance(json_response, list) and len(json_response) > 0:
                coords = json_response[0].get("point_2d")
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    return [int(float(x)) for x in coords]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numbers with regex
        numbers = re.findall(r"-?\d+", response.split("point_2d")[-1])
        if len(numbers) >= 2:
            return [int(numbers[0]), int(numbers[1])]

        return None

        
class UITARSPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        user_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

        messages = [
            {
                "role": "user",
                "content": f"<image>'{user_prompt.format(instruction=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        # Try <point>x y</point>
        m = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", response)
        if m:
            return [int(m.group(1)), int(m.group(2))]

        # Try JSON { "coordinate": [x, y] }
        try:
            data = json.loads(re.search(r"\{.*\}", response).group(0))
            coords = data.get("coordinate") or data.get("point")
            if coords and len(coords) == 2:
                return [int(coords[0]), int(coords[1])]
        except:
            pass

        # Fallback: first two numbers
        nums = re.findall(r"\d+", response)
        return [int(nums[0]), int(nums[1])] if len(nums) >= 2 else None

class UIR1EPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        
        user_prompt = (
        "In this UI screenshot, I want to perform the command '{instruction}'.\n"
        "Please provide the action to perform (enumerate in ['click'])"
        "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
        "Output the final answer in <answer> </answer> tags directly."
        "The output answer format should be as follows:\n"
        "<answer>[{{'action': 'click', 'coordinate': [x, y]}}]</answer>\n"
        "Please strictly follow the format."
    )


        messages = [
            {
                "role": "user",
                "content": f"<image>\n'{user_prompt.format(instruction=sample['instruction'])}'"
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        # Extract content inside <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()

        # Try JSON/dict parsing
        try:
            # Ensure valid JSON (replace single quotes with double quotes)
            data = json.loads(content.replace("'", '"'))
            if isinstance(data, list) and "coordinate" in data[0]:
                coords = data[0]["coordinate"]
                if len(coords) == 2:
                    return [int(coords[0]), int(coords[1])]
        except:
            pass

        # Fallback: just grab two numbers
        nums = re.findall(r"\d+", content)
        return [int(nums[0]), int(nums[1])] if len(nums) >= 2 else None


class GUIActorPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:


        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample['instruction']
                    },
                ],
            },
        ]

        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        # Extract content inside <answer>...</answer>
        click_point = json.loads(response)['predicted_coords']
        return click_point
        
class PhiGroundPrompt(BasicPrompt):
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:

        system_prompt = """
The description of the element: 
{RE}


Locate the above described element in the image. The output should be bounding box using relative coordinates multiplying 1000.
"""

        messages = [

            {
                "role": "user",
                "content": system_prompt.format(RE=sample['instruction'])
            },
        ]

        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List[int]]:
        try:
            pred = response.strip().split("<point>")[1].split("</point>")[0]
            coors = [float(o) / 1000 for o in pred.split(", ")]
            assert len(coors) == 2
        except:
            coors = None

        return coors

    
# Register available prompt processors
PROMPT_PROCESSORS = {
    "infigui-g1": InfiguiG1Prompt,
    "groundcua": GroundCUAPrompt,
    "infigui-r1": InfiguiR1Prompt,
    "gta1": GTAPrompt,
    "opencua": OpenCUAPrompt,
    "jedi": BasicPrompt, # same as system prompt, with "wait" action
    "guig2": GUIG2Prompt,
    "guig1": GUIG1Prompt, # Yuqi-Zhou/GUI-G1-3B-v1
    "uground-v1": UGroundV1Prompt,
    "segui": SEGUIPrompt,
    "uitars": UITARSPrompt,
    "uir1e": UIR1EPrompt,
    "guiactor": GUIActorPrompt,
    "phiground": PhiGroundPrompt,
}


def get_prompt_processor(name: str):
    """
    Retrieve a prompt processor by name.
    """
    if name not in PROMPT_PROCESSORS:
        raise ValueError(f"Unknown prompt processor: {name}. Available: {list(PROMPT_PROCESSORS.keys())}")
    return PROMPT_PROCESSORS[name]