"""
Utility functions for working with GroundCUA instruction tuning data.
This can be used with spatial_data.json and functional_instructions_extra.json
"""

import os
import json
import math


# Update this path to point to your local GroundCUA repository
GROUNDCUA_DATA_PATH = '/path/to/GroundCUA/data'


def find_element_by_coordinate(platform: str, element_id: str, coordinate: list, data_path: str = None) -> dict | None:
    """
    Find the element whose bounding box contains the given coordinate.
    If multiple elements contain the coordinate, return the one whose 
    bounding box center is closest to the given coordinate.
    
    Args:
        platform: The platform name (e.g., 'Affine')
        element_id: The unique identifier for the screenshot (the 'id' field from instruction data)
        coordinate: [x, y] coordinate (center point)
        data_path: Path to GroundCUA data directory (optional, uses GROUNDCUA_DATA_PATH if not provided)
    
    Returns:
        The matching element dict, or None if not found
    """
    if data_path is None:
        data_path = GROUNDCUA_DATA_PATH
    
    json_path = os.path.join(data_path, platform, element_id + '.json')
    
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        elements = json.load(f)
    
    x, y = coordinate
    
    # Find all elements whose bounding box contains the coordinate
    matching_elements = []
    for element in elements:
        bbox = element['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        # Normalize bbox coordinates (ensure x1 < x2 and y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Check if coordinate is within this bounding box
        if x1 <= x <= x2 and y1 <= y <= y2:
            # Calculate the center of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate distance from the given coordinate to the bbox center
            distance = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
            
            matching_elements.append((element, distance))
    
    if not matching_elements:
        return None
    
    # Sort by distance and return the element with the closest center
    matching_elements.sort(key=lambda item: item[1])
    return matching_elements[0][0]


def find_element_for_instruction(instance: dict, data_path: str = None) -> dict | None:
    """
    Given an instruction instance, find and return the full element dict
    including bbox, text, category, etc.
    
    Args:
        instance: Dict with 'id', 'platform', 'coordinate' keys
        data_path: Path to GroundCUA data directory (optional)
    
    Returns:
        The full element dict or None if not found
    """
    # Handle both 'coordinate' and 'cordinate' spellings for backwards compatibility
    coord_key = 'coordinate' if 'coordinate' in instance else 'cordinate'
    
    return find_element_by_coordinate(
        platform=instance['platform'],
        element_id=instance['id'],
        coordinate=instance[coord_key],
        data_path=data_path
    )