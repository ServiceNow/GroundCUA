#!/usr/bin/env python3
"""
Generate task list JSON pointing to CoT trajectory files for downstream use.

Usage:
    python generate_task_list.py --data_dir ./videocua_processed --output task_list_cot.json
    python generate_task_list.py --data_dir ./videocua_processed --output task_list_cot.json --model_suffix anthropic-claude-sonnet-4.5_cot_v1
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional


def find_cot_trajectories(data_dir: str, model_suffix: Optional[str] = None) -> List[Dict]:
    """
    Find all CoT trajectory files in the data directory.

    Args:
        data_dir: Root data directory containing app folders
        model_suffix: Optional model suffix to filter (e.g., 'anthropic-claude-sonnet-4.5_cot_v1')

    Returns:
        List of task info dictionaries
    """
    data_path = Path(data_dir).resolve()
    tasks = []

    # Pattern: data_dir/AppName/task_id/model_suffix/{task_id_cot.jsonl OR opencua_trace_with_cot.jsonl}
    for cot_file in data_path.rglob("*_cot.jsonl"):
        try:
            model_folder = cot_file.parent
            task_folder = model_folder.parent
            app_folder = task_folder.parent

            task_id = task_folder.name
            app_name = app_folder.name

            # Skip if model_suffix is specified and doesn't match
            if model_suffix and model_suffix not in model_folder.name:
                continue

            # Accept both filename formats:
            # 1. {task_id}_cot.jsonl (legacy format)
            # 2. opencua_trace_with_cot.jsonl (gen_cot.py output)
            expected_filename_legacy = f"{task_id}_cot.jsonl"
            expected_filename_new = "opencua_trace_with_cot.jsonl"

            if cot_file.name != expected_filename_legacy and cot_file.name != expected_filename_new:
                print(f"Warning: Skipping {cot_file} - unexpected filename format")
                continue

            task_info = {
                "task_id": task_id,
                "task_record": {
                    "path": str(task_folder),
                    "app": app_name
                },
                "traj_path": str(cot_file)
            }
            tasks.append(task_info)

        except Exception as e:
            print(f"Warning: Error processing {cot_file}: {e}")
            continue

    # Sort by app name and task_id for consistent ordering
    tasks.sort(key=lambda x: (x["task_record"]["app"], x["task_id"]))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate task list JSON from CoT trajectory files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root data directory containing app folders with task trajectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default=None,
        help="Filter by model suffix in folder name"
    )

    args = parser.parse_args()

    print(f"Scanning {args.data_dir} for CoT trajectory files...")
    if args.model_suffix:
        print(f"Filtering by model suffix: {args.model_suffix}")

    tasks = find_cot_trajectories(args.data_dir, args.model_suffix)

    if not tasks:
        print("No CoT trajectory files found!")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"\nFound {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task['task_record']['app']}/{task['task_id']}")

    print(f"\nTask list saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
