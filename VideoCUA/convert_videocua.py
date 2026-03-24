#!/usr/bin/env python3
"""
Convert VideoCUA data format to trace.jsonl format required by gen_cot.py.

VideoCUA format (action_log.json):
{
    "task_id": 46551,
    "task_instruction": "Switch to Wireframe mode...",
    "platform": "Blender",
    "action_log": [
        {"action_type": "CLICK", "timestamp": 1.2, "action_params": {"x": 1748, "y": 95, ...}},
        {"action_type": "AFTER_LAST_ACTION", "timestamp": 3.362, ...}
    ]
}

Output format (trace.jsonl):
{
    "task_id": "46551",
    "instruction": "On this Blender platform, please Switch to Wireframe mode...",
    "traj": [
        {"index": 0, "image": "/path/to/processed_images/0.png", "value": {"code": "pyautogui.click(...)"}},
        ...
    ]
}

Usage:
    # Convert all tasks in a directory
    python convert_videocua.py --data_dir VideoCUA/data --output_dir videocua_processed

    # Convert specific platforms only
    python convert_videocua.py --data_dir VideoCUA/data --output_dir videocua_processed --platforms "Blender,GIMP"
"""

import argparse
import json
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import cv2
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil


def extract_frames_from_video(video_path: str, output_dir: str, timestamps: List[float]) -> List[str]:
    """
    Extract frames from video at specified timestamps.

    Returns list of saved frame paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_paths = []

    for i, ts in enumerate(timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(output_dir, f"{i}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        else:
            # Try to get closest frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(output_dir, f"{i}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            else:
                frame_paths.append(None)

    cap.release()
    return frame_paths


def get_video_resolution(video_path: str) -> tuple:
    """Get video resolution (width, height)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)


def action_to_code(action: Dict, resolution: tuple) -> str:
    """
    Convert action dict to pyautogui code string.
    Coordinates are normalized to 0-1 range.
    """
    action_type = action.get("action_type", "").upper()
    params = action.get("action_params", {})

    width, height = resolution

    def fmt_xy(x, y):
        norm_x = x / width if x is not None else None
        norm_y = y / height if y is not None else None
        return f"{norm_x:.4f}" if norm_x is not None else "None", f"{norm_y:.4f}" if norm_y is not None else "None"

    if action_type in ("TERMINATE_SUCCESS", "TERMINATE", "AFTER_LAST_ACTION"):
        return "computer.terminate(status='success')"

    if action_type == "CLICK":
        x, y = params.get("x"), params.get("y")
        nx, ny = fmt_xy(x, y)
        key = params.get("text", "left").lower() if params.get("text") in ["Left", "Right", "Middle"] else "left"
        num_clicks = params.get("numClicks") or 1

        if key == "right":
            return f"pyautogui.rightClick(x={nx}, y={ny})"
        if num_clicks >= 2:
            return f"pyautogui.doubleClick(x={nx}, y={ny})"
        return f"pyautogui.click(x={nx}, y={ny})"

    if action_type == "MOVE_TO":
        x, y = params.get("x"), params.get("y")
        nx, ny = fmt_xy(x, y)
        return f"pyautogui.moveTo(x={nx}, y={ny})"

    if action_type == "DRAG_TO":
        x, y = params.get("x"), params.get("y")
        nx, ny = fmt_xy(x, y)
        button = params.get("button", "left").lower()
        return f"pyautogui.dragTo(x={nx}, y={ny}, button='{button}')"

    if action_type == "SCROLL":
        amount = params.get("scrollY", params.get("scroll", 0))
        return f"pyautogui.scroll({amount})"

    if action_type in ("TYPE", "TYPING", "TEXT"):
        text = params.get("text", "")
        return f"pyautogui.typewrite({json.dumps(text)})"

    if action_type == "PRESS":
        key = params.get("key", params.get("text", ""))
        return f"pyautogui.press({json.dumps(key)})"

    if action_type == "HOTKEY":
        keys = params.get("keys", [])
        if not keys:
            key_str = params.get("key", params.get("text", ""))
            keys = [k.strip() for k in key_str.split("+") if k.strip()]
        keys_list = ", ".join(json.dumps(k) for k in keys)
        return f"pyautogui.hotkey({keys_list})"

    if action_type == "KEY_DOWN":
        key = params.get("key", params.get("text", ""))
        return f"pyautogui.keyDown({json.dumps(key)})"

    if action_type == "KEY_UP":
        key = params.get("key", params.get("text", ""))
        return f"pyautogui.keyUp({json.dumps(key)})"

    if action_type == "MOUSE_DOWN":
        button = params.get("button", "left").lower()
        return f"pyautogui.mouseDown(button='{button}')"

    if action_type == "MOUSE_UP":
        button = params.get("button", "left").lower()
        return f"pyautogui.mouseUp(button='{button}')"

    return f"# Unsupported action: {action_type}"


def convert_task(task_dir: str, output_base_dir: Optional[str] = None) -> Dict:
    """
    Convert a single VideoCUA task to trace.jsonl format. (OpenCUA style trace with code values)

    Args:
        task_dir: Path to task directory containing action_log.json and video/
        output_base_dir: If provided, copy processed data here. Otherwise, process in-place.

    Returns:
        Task info dict with task_id, task_record, and traj_path
    """
    task_dir = Path(task_dir)
    task_id = task_dir.name
    platform = task_dir.parent.name

    # Determine output directory
    if output_base_dir:
        output_dir = Path(output_base_dir) / platform / task_id
    else:
        output_dir = task_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load action log
    action_log_path = task_dir / "action_log.json"
    if not action_log_path.exists():
        raise FileNotFoundError(f"action_log.json not found in {task_dir}")

    with open(action_log_path, 'r') as f:
        data = json.load(f)

    task_instruction = data.get("task_instruction", "")
    platform_name = data.get("platform", platform)
    action_log = data.get("action_log", [])

    # Find video file
    video_dir = task_dir / "video"
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in (".mp4", ".avi", ".mov", ".webm")]
    if not video_files:
        raise FileNotFoundError(f"No video file found in {video_dir}")
    video_path = str(video_files[0])

    # Get resolution
    resolution = get_video_resolution(video_path)

    # Extract timestamps (use timestamp before each action)
    timestamps = []
    for action in action_log:
        ts = action.get("timestamp", 0)
        timestamps.append(max(0, ts - 0.1))  # Slightly before action

    # Extract frames
    processed_images_dir = output_dir / "processed_images"
    frame_paths = extract_frames_from_video(video_path, str(processed_images_dir), timestamps)

    # Build trajectory
    instruction = f"On this {platform_name} platform, please {task_instruction}"

    traj = []
    for i, (action, frame_path) in enumerate(zip(action_log, frame_paths)):
        if frame_path is None:
            continue

        code = action_to_code(action, resolution)
        # Use absolute path for image
        abs_frame_path = str(Path(frame_path).resolve())
        traj.append({
            "index": i,
            "image": abs_frame_path,
            "value": {"code": code}
        })

    # Create output
    output_data = {
        "task_id": str(task_id),
        "instruction": instruction,
        "traj": traj
    }

    # Save to jsonl
    traj_path = output_dir / "trace.jsonl"
    with open(traj_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False)
        f.write("\n")

    # Copy action_log.json to output if different location
    if output_base_dir and output_dir != task_dir:
        shutil.copy2(action_log_path, output_dir / "action_log.json")

    return {
        "task_id": str(task_id),
        "task_record": {
            "app": platform_name,
            "path": str(output_dir.resolve())
        },
        "traj_path": str(traj_path.resolve())
    }


def convert_task_wrapper(args):
    """Wrapper for multiprocessing."""
    task_dir, output_base_dir = args
    try:
        return convert_task(task_dir, output_base_dir), None
    except Exception as e:
        return None, f"Error processing {task_dir}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert VideoCUA data format to trace.jsonl format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to VideoCUA/data directory containing platform folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--platforms",
        type=str,
        default=None,
        help="Comma-separated list of platforms to process (default: all)"
    )
    parser.add_argument(
        "--task_list_output",
        type=str,
        default=None,
        help="Output path for task list JSON (default: output_dir/task_list.json)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory {data_path} does not exist")
        return 1

    # Collect tasks to convert
    tasks_to_convert = []
    platforms_filter = set(args.platforms.split(",")) if args.platforms else None

    for platform_dir in sorted(data_path.iterdir()):
        if not platform_dir.is_dir():
            continue
        if platforms_filter and platform_dir.name not in platforms_filter:
            continue

        for task_dir in platform_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if (task_dir / "action_log.json").exists():
                tasks_to_convert.append(str(task_dir))

    print(f"Found {len(tasks_to_convert)} tasks to convert")

    # Convert tasks
    task_list = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(convert_task_wrapper, (task_dir, args.output_dir))
            for task_dir in tasks_to_convert
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            result, error = future.result()
            if error:
                errors.append(error)
            elif result:
                task_list.append(result)

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for e in errors:
            print(f"  - {e}")

    # Save task list
    task_list_path = args.task_list_output or os.path.join(args.output_dir, "task_list.json")
    os.makedirs(os.path.dirname(task_list_path) or ".", exist_ok=True)

    with open(task_list_path, 'w') as f:
        json.dump(task_list, f, indent=2)

    print(f"\nSuccessfully converted {len(task_list)} tasks")
    print(f"Task list saved to: {task_list_path}")

    return 0


if __name__ == "__main__":
    exit(main())
