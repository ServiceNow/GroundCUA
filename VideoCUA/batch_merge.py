#!/usr/bin/env python3
"""
Batch merge script for CoT results across multiple tasks and platforms.

Usage:
    python batch_merge.py \
        --data_dir actcua_processed \
        --model_folder anthropic-claude-sonnet-4.5_cot_v1 \
        --output_file final_cot_results.jsonl
"""

# Ensure imports resolve correctly regardless of working directory
import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from merge_json import merge_json_to_jsonl, process_subdir


def find_cot_tasks_folders(data_dir: str, model_folder: str) -> list:
    """
    Find all cot_tasks folders matching the model_folder pattern.

    Args:
        data_dir: Root data directory
        model_folder: Model folder name or prefix

    Returns:
        List of tuples: (platform_name, task_id, cot_tasks_path, actual_folder_name)
    """
    data_path = Path(data_dir)
    results = []

    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return results

    for platform_dir in sorted(data_path.iterdir()):
        if not platform_dir.is_dir():
            continue

        platform_name = platform_dir.name

        for task_dir in sorted(platform_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_id = task_dir.name

            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                if model_dir.name == model_folder or model_dir.name.startswith(f"{model_folder}_"):
                    cot_tasks_path = model_dir / "cot_tasks"
                    if cot_tasks_path.exists() and cot_tasks_path.is_dir():
                        results.append((platform_name, task_id, str(cot_tasks_path), model_dir.name))

    return results


def merge_single_task(cot_tasks_path: str, output_file: str = None) -> str:
    """
    Merge a single task's CoT results.

    Returns:
        Path to the merged JSONL file, or None if failed
    """
    try:
        cot_tasks_path = str(Path(cot_tasks_path).resolve())
        if output_file:
            output_file = str(Path(output_file).resolve())

        merged_path = merge_json_to_jsonl(
            input_dir=cot_tasks_path,
            output_file=output_file,
            use_multiprocessing=False
        )
        return merged_path
    except Exception as e:
        logger.error(f"Failed to merge {cot_tasks_path}: {e}")
        return None


def merge_all_jsonl_files(jsonl_files: list, output_file: str) -> str:
    """
    Merge multiple JSONL files into a single file.
    """
    all_records = []

    for jsonl_path in tqdm(jsonl_files, desc="Reading JSONL files"):
        if not os.path.exists(jsonl_path):
            logger.warning(f"JSONL file not found: {jsonl_path}")
            continue

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line in {jsonl_path}: {e}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Merged {len(all_records)} records into {output_path}")
    return str(output_path)


def batch_merge(
    data_dir: str,
    model_folder: str,
    output_file: str = None,
    skip_existing: bool = True
) -> str:
    """
    Batch merge all CoT results from a data directory.
    """
    logger.info(f"Scanning {data_dir} for model folder: {model_folder}")
    cot_folders = find_cot_tasks_folders(data_dir, model_folder)

    if not cot_folders:
        logger.error(f"No cot_tasks folders found for model: {model_folder}")
        return None

    logger.info(f"Found {len(cot_folders)} tasks with CoT results")

    platforms = {}
    for platform, task_id, path, actual_folder in cot_folders:
        if platform not in platforms:
            platforms[platform] = []
        platforms[platform].append(task_id)

    logger.info("Summary by platform:")
    for platform, tasks in sorted(platforms.items()):
        logger.info(f"  {platform}: {len(tasks)} tasks")

    merged_files = []
    failed_tasks = []

    for platform, task_id, cot_path, actual_folder in tqdm(cot_folders, desc="Merging tasks"):
        task_output = Path(cot_path).resolve().parent / f"{task_id}_cot.jsonl"

        if skip_existing and task_output.exists():
            logger.debug(f"Skipping existing: {task_output}")
            merged_files.append(str(task_output))
            continue

        merged_path = merge_single_task(cot_path, str(task_output))
        if merged_path:
            merged_files.append(merged_path)
        else:
            failed_tasks.append((platform, task_id))

    if failed_tasks:
        logger.warning(f"Failed to merge {len(failed_tasks)} tasks:")
        for platform, task_id in failed_tasks:
            logger.warning(f"  {platform}/{task_id}")

    if output_file is None:
        output_file = f"{model_folder}_all_cot.jsonl"

    output_path = Path(output_file)

    logger.info(f"Merging {len(merged_files)} task files into final output...")
    final_path = merge_all_jsonl_files(merged_files, str(output_path))

    logger.info(f"=" * 60)
    logger.info(f"Batch merge complete!")
    logger.info(f"  Total tasks processed: {len(cot_folders)}")
    logger.info(f"  Successfully merged: {len(merged_files)}")
    logger.info(f"  Failed: {len(failed_tasks)}")
    logger.info(f"  Final output: {final_path}")
    logger.info(f"=" * 60)

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch merge CoT results from multiple tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_merge.py --data_dir actcua_processed --model_folder anthropic-claude-sonnet-4.5_cot_v1

  python batch_merge.py --data_dir actcua_processed --model_folder anthropic-claude-sonnet-4.5_cot_v1 --output_file results/all_cot.jsonl

  python batch_merge.py --data_dir actcua_processed --model_folder anthropic-claude-sonnet-4.5_cot_v1 --no_skip_existing
        """
    )

    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--model_folder", type=str, required=True, help="Model folder name or prefix")
    parser.add_argument("--output_file", type=str, default=None, help="Final output JSONL file path")
    parser.add_argument("--no_skip_existing", action="store_true", help="Re-merge tasks even if merged JSONL already exists")

    args = parser.parse_args()

    batch_merge(
        data_dir=args.data_dir,
        model_folder=args.model_folder,
        output_file=args.output_file,
        skip_existing=not args.no_skip_existing
    )


if __name__ == "__main__":
    main()
