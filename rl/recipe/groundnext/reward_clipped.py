# Copyright 2025 Individual Contributor: InfiX.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import re
import numpy as np
from itertools import combinations

FMT_RATIO = 0
ACC_RATIO = 1


# ============================================================================
# Utility Functions
# ============================================================================


def point_to_bbox_distance(pred_coords, ground_truth, image_size):
    px, py = pred_coords
    x_min, y_min, x_max, y_max = ground_truth['x1'], ground_truth['y1'], ground_truth['x2'], ground_truth['y2']
    img_w, img_h = image_size
    
    assert x_min < x_max
    assert y_min < y_max

    # If inside the box

    if x_min <= px <= x_max and y_min <= py <= y_max:
        bbox_diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        if bbox_diag == 0:
            return 0.0
            # Max distance to the corners of the bbox
        corners = [
            (x_min, y_min), (x_min, y_max),
            (x_max, y_min), (x_max, y_max),
        ]
        min_corner_dist = min(
            np.sqrt((px - cx)**2 + (py - cy)**2) for cx, cy in corners
        )
        # Normalize the distance to the bbox diagonal
        
        norm_min_corner_dist = (2 * min_corner_dist) / bbox_diag if bbox_diag > 0 else 0.0
        return {"norm": norm_min_corner_dist,
                "raw": min_corner_dist}
    
    # Clamp the point to the bbox edges
    clamped_x = np.clip(px, x_min, x_max)
    clamped_y = np.clip(py, y_min, y_max)

    # Compute Euclidean distance to the clamped point
    dist_2_bbox = np.sqrt((px - clamped_x)**2 + (py - clamped_y)**2)
    
    # Distance from image corners to nearest point on bbox
    image_corners = {
        "top_left": (0, 0),
        "top_right": (img_w, 0),
        "bottom_left": (0, img_h),
        "bottom_right": (img_w, img_h),
    }

    corner_to_bbox_dists = {}
    max_dist = 0
    for name, (ix, iy) in image_corners.items():
        clamped_ix = np.clip(ix, x_min, x_max)
        clamped_iy = np.clip(iy, y_min, y_max)
        dist = np.sqrt((ix - clamped_ix)**2 + (iy - clamped_iy)**2)
        corner_to_bbox_dists[name] = dist
        if max_dist < dist:
            max_dist = dist
    
    norm_pred_2_bbox = dist_2_bbox / max_dist if max_dist > 0 else 0.0

    return {"norm": -1 * norm_pred_2_bbox,
            "raw": -1 * dist_2_bbox}

def extract_coord(content):
    """
    Extract the [x, y] coordinate from a <tool_call> JSON block.
    If not found, return [0, 0].
    """
    # Try to find the JSON inside <tool_call> tags
    answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)

    if not content_answer_match:
        return [0, 0], False

    json_str = content_answer_match.group(1).strip()
    try:
        obj = json.loads(json_str)
        coord = obj.get("arguments", {}).get("coordinate", [0, 0])
        if (
            isinstance(coord, list)
            and len(coord) == 2
            and all(isinstance(i, int) for i in coord)
        ):
            return coord, True
        else:
            return [0, 0], False
    except json.JSONDecodeError:
        return [0, 0], False
    



def extract_think_format(predict_str: str) -> None | dict[str, str]:
    """
    Check if the predicted string meets format requirements and extract thinking and answer parts.

    Args:
        predict_str: The predicted string

    Returns:
        If format requirements are met, returns a dictionary containing thinking and answer parts;
        otherwise returns None
    """
    if not predict_str or not isinstance(predict_str, str):
        return None

    # Check if <think> is at the beginning
    if not predict_str.startswith("<think>"):
        return None

    # Check if there is <think>...</think> format
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, predict_str, re.DOTALL)
    if not think_match:
        return None

    if predict_str.count("<think>") != 1 or predict_str.count("</think>") != 1:
        return None

    # Extract thinking content
    think_content = think_match.group(1).strip()
    if not think_content:
        return None

    # Get content after </think>
    think_end_pos = predict_str.find("</think>") + len("</think>")
    post_think_content = predict_str[think_end_pos:].strip()

    # Check if there is non-empty content after </think>
    if not post_think_content:
        return None

    return {"think": think_content, "answer": post_think_content}


def extract_and_parse_json(input_string, wrapper):
    """
    Try to extract and parse JSON from a string.

    Args:
        input_string: The input string
        wrapper: JSON wrapper symbols, can be '{}' or '[]'

    Returns:
        Parsed JSON object, returns None if parsing fails
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)

    if start_index == -1:
        return None

    # Find the matching end character by balancing brackets/braces
    balance = 1
    end_index = -1
    for i in range(start_index + 1, len(input_string)):
        if input_string[i] == start_char:
            balance += 1
        elif input_string[i] == end_char:
            balance -= 1

        if balance == 0:
            end_index = i
            break

    if end_index == -1:
        return None

    json_string = input_string[start_index : end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None

def _format_reward(answer):
    # Ensure exactly one <tool_call> and one </tool_call>
    if answer.count("<tool_call>") != 1 or answer.count("</tool_call>") != 1:
        return 0.0

    
    answer_match = re.search(r"<tool_call>(.*?)</tool_call>", answer, re.DOTALL)
    
    if not answer_match:
        return 0.0

    # 提取 <answer> 内的内容并解析为 JSON 格式
    answer_content = answer_match.group(1).strip()
    # ic(answer_content)
    
    try:
        actions = eval(answer_content)
        # ic(actions)

        if not isinstance(actions, list) and \
            isinstance(actions, dict):
            actions = [actions]

        # 验证每个 action 的格式
        for action in actions:
            # ic(action)
            if not isinstance(action, dict):
                return 0.0
            
            if "name" not in action or "arguments" not in action:
                return 0.0
            
            
            if "action" not in action['arguments'] or "coordinate" not in action['arguments']:
                return 0.0
            
            if not isinstance(action["name"], str):
                return 0.0
            
            if action["name"] != 'computer_use':
                return 0.0
            
            if not (isinstance(action['arguments']['coordinate'][0],int) and isinstance(action['arguments']['coordinate'][1],int)): 
                return 0.0
        
        return 1.0
    
    except:
        return 0.0


def _accuracy_reward(answer, ground_truth, image_size):
    """
    predict_str, ground_truth, image_size
    """
    if ground_truth['x1'] > ground_truth['x2']:
        ground_truth['x1'], ground_truth['x2'] = ground_truth['x2'], ground_truth['x1']
        
    if ground_truth['y1'] > ground_truth['y2']:
        ground_truth['y1'], ground_truth['y2'] = ground_truth['y2'], ground_truth['y1']

    max_distance = np.sqrt(image_size[0]**2 + image_size[1]**2)
    accuracy = 0.0
    pred_coords = [-1, -1]
    dist = {"norm": -1,
            "raw": -1 * max_distance}
        
    try:
        pred_coords, _ = extract_coord(answer)
        
        if (ground_truth['x1'] < pred_coords[0] < ground_truth['x2']) and (ground_truth['y1'] <pred_coords[1] <ground_truth['y2']):
            accuracy = 1.0
        else:
            accuracy = 0.0
            
        # pred_coords, _ = extract_coord(answer)
        dist = point_to_bbox_distance(pred_coords, ground_truth, image_size)
        
    except Exception as e:
        print(e)
        
    
    return accuracy, dist, json.dumps(pred_coords)
    

def calculate_point_reward(answer, ground_truth, extra_info=None, fmt_ratio=1.0, acc_ratio=1.0, reward_threshs_string=['0.1', '0.5', '1.0'], **kwargs):
    """
    Calculate the final reward for 'point' type data.

    Implements the full logic including format checks, collinearity checks,
    and the zero-centered symmetric reward calculation.

    Args:
        answer: The solution string from the model
        ground_truth: Ground truth data
        extra_info: Extra information dictionary
        fmt_ratio: Format reward ratio
        acc_ratio: Accuracy reward ratio
        **kwargs: Additional keyword arguments

    Returns:
        Dictionary containing detailed reward information
    """

    # Reuse _format_reward to check the format of the 'answer' part
    # If it's invalid, return score of -1
    
    
    reward_threshs = []
    for r in reward_threshs_string:
        if isinstance(r, str):
            try:
                r_float = float(r)
                reward_threshs.append(r_float)
            except:
                raise ValueError(f"reward value {r} must be convertible to float")
        elif isinstance(r, (int, float)):
            reward_threshs.append(float(r))
        else:
            raise ValueError(f"reward value {r} must be str, int or float")
    
    if len(reward_threshs) == 0:
        raise ValueError("rewards list must not be empty")
    
    for r in reward_threshs: # e.g. rewards = [1, 0.5, 0.1]
        assert r > 0, "All reward values must be positive"
        
    
    # append negative rewards to the beginning
    # e.g. if rewards = [1, 0.5, 0.1] -> [-1, -0.5, -0.1, 0.1, 0.5, 1]
    reward_threshs = [-r for r in reversed(reward_threshs)] + reward_threshs
    
    assert max(reward_threshs) == 1
    
    reward_threshs = sorted(reward_threshs) # e.g., ensure sorted order [-1, -0.5, -0.1, 0.1, 0.5, 1]
    
    
    format_reward = _format_reward(answer)


    if format_reward == 0.0:
        format_reward = -1.0
    # If format is OK, calculate the accuracy reward
    accuracy_reward, dist_reward, extracted_answer = _accuracy_reward(answer, ground_truth, image_size=extra_info['image_size'])
    
    norm_dist = dist_reward['norm']
    raw_dist = dist_reward['raw']
    
    thresholds = reward_threshs[1:-1]  # Exclude the extreme values for threshold checks e.g. [-0.5, -0.1, 0.1, 0.5]
    clipped_reward = -10.0
    # Clipped reward based on normalized distance
    
    if len(thresholds) == 0:
        if norm_dist > 0:
            clipped_reward = reward_threshs[-1] # e.g. 1.0
        else:
            clipped_reward = reward_threshs[0] # e.g. -1.0
    
    else:
        if norm_dist >= thresholds[-1]:
            clipped_reward = reward_threshs[-1] # e.g. 1.0
        elif norm_dist < thresholds[0]:
            clipped_reward = reward_threshs[0] # e.g. -1.0

        else: # e.g. reward_threshs = [-1, -0.5, -0.1, 0.1, 0.5, 1], thresholds = [-0.5, -0.1, 0.1, 0.5]
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= norm_dist < thresholds[i + 1]:
                    if norm_dist < 0:
                        clipped_reward = thresholds[i]
                    else:
                        clipped_reward = thresholds[i + 1]
                    break
                    

    return {
        "score": clipped_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer,
        "raw_distance": raw_dist,
        "norm_distance": norm_dist,
        "clipped_reward": clipped_reward
    }


# ============================================================================
# AER Reward Handler Registry
# ============================================================================

# Dictionary to map data_source to the respective reward calculation function
AER_REWARD_HANDLERS = {
    "point": calculate_point_reward,
}


def gui_reward_function(data_source, solution_str, ground_truth, extra_info=None, **kwargs):

    handler = AER_REWARD_HANDLERS.get(data_source, None)

    if handler:
        try:
            return handler(
                solution_str, ground_truth, extra_info=extra_info, fmt_ratio=FMT_RATIO, acc_ratio=ACC_RATIO, **kwargs
            )
        except Exception as e:
            logging.exception(
                f"Error executing reward handler for data_source '{data_source}': {e}",
            )
            return {
                "score": -1.0,
                "format": -1.0,
                "accuracy": -1.0,
                "pred": "",
                "raw_distance": -1.0,
                "norm_distance": -1.0,
                "clipped_reward": -1.0
            }  # Return a default penalty score on error
    else:
        raise ValueError(f"Unknown data_source: '{data_source}'. No specific reward handler defined.")


