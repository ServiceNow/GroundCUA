import json
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from loguru import logger

def process_subdir(subdir):
    """
    Process a single subdirectory to merge meta.json and step files.

    Args:
        subdir (Path): Path to the subdirectory containing JSON files

    Returns:
        dict or None: Merged data if successful, None if failed
    """
    meta_file = subdir / "meta.json"
    if not meta_file.exists():
        return None

    try:
        with open(meta_file, "r", encoding='utf-8') as f:
            meta_data = json.load(f)

        traj = []
        json_files = sorted(
            [f for f in subdir.glob("*.json") if f.name != "meta.json"],
            key=lambda x: int(x.stem) if x.stem.isdigit() else float("inf")
        )

        for jf in json_files:
            with open(jf, "r", encoding='utf-8') as f:
                traj.append(json.load(f))

        meta_data["traj"] = traj
        return meta_data

    except Exception as e:
        logger.error(f"Error processing {subdir}: {e}")
        return None


def merge_json_to_jsonl(input_dir, output_file=None, use_multiprocessing=True):
    """
    Merge JSON files from subdirectories into a single JSONL file.

    Args:
        input_dir (str): Input directory containing subdirectories with JSON files
        output_file (str, optional): Output JSONL file path. If None, creates in parent directory
        use_multiprocessing (bool): Whether to use multiprocessing for faster processing

    Returns:
        str: Path to the generated JSONL file
    """
    input_dir = Path(input_dir)

    if output_file is None:
        output_file = "traj_with_cot_example.jsonl"

    # If output_file is just a filename, put it in the parent directory of input_dir
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = input_dir.parent / output_file

    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(subdirs)} task directories to process...")

    output_lines = []

    if use_multiprocessing and len(subdirs) > 1:
        with Pool(cpu_count()) as pool:
            for result in tqdm(
                pool.imap_unordered(process_subdir, subdirs),
                total=len(subdirs),
                desc="Processing tasks"
            ):
                if result is not None:
                    output_lines.append(result)
    else:
        for subdir in tqdm(subdirs, desc="Processing tasks"):
            result = process_subdir(subdir)
            if result is not None:
                output_lines.append(result)

    logger.info(f"Successfully processed {len(output_lines)} / {len(subdirs)} tasks...")

    with open(output_path, "w", encoding='utf-8') as f:
        for item in output_lines:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Merge finished, saved to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge JSON files into a single JSONL file.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing JSON files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output path for the merged JSONL file."
    )
    parser.add_argument(
        "--no_multiprocessing",
        action='store_true',
        help="Disable multiprocessing for debugging."
    )

    args = parser.parse_args()

    merge_json_to_jsonl(
        input_dir=args.input_dir,
        output_file=args.output_file,
        use_multiprocessing=not args.no_multiprocessing
    )


if __name__ == "__main__":
    main()
