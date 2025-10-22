import argparse
import os
import json
import random
from datetime import datetime
from typing import Dict
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import smart_resize

# Import custom modules

from data import (
    load_benchmark_info, 
    load_dataset, 
    standardize_sample,
    resize_image,
    calculate_hierarchical_statistics,
    print_hierarchical_stats,
    reorder_stats_for_output
)
from prompts import get_prompt_processor, PROMPT_PROCESSORS

def prepare_sample_data_phiground(sample: Dict, benchmark_info: Dict, max_image_pixels: int, prompt_processor, think_mode: bool, num_crops=7) -> tuple:
    
    image_path = os.path.join(benchmark_info["image_root"], sample["images"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Process bounding box coordinates
    bbox = sample["bbox"].copy()
    

    if num_crops == 16:
        target_width, target_height = 336 * 5, 336 *3
    elif num_crops == 7:
        target_width, target_height = 336 * 3, 336 *2
    elif num_crops == 28:
        target_width, target_height = 336 * 7, 336 *4
    else:
        raise NotImplementedError

    img_ratio = original_width / original_height
    target_ratio = target_width / target_height
   
    if img_ratio > target_ratio:  
        new_width = target_width  
        new_height = int(new_width / img_ratio)
    else:  
        new_height = target_height
        new_width = int(new_height * img_ratio)  
        
    reshape_ratio = new_width / original_width
    
    if all(key in bbox.keys() for key in ["x1", "y1", "x2", "y2"]):
        bbox["x1"] = bbox["x1"] * reshape_ratio
        bbox["y1"] = bbox["y1"] * reshape_ratio
        bbox["x2"] = bbox["x2"] * reshape_ratio
        bbox["y2"] = bbox["y2"] * reshape_ratio
    elif all(key in bbox.keys() for key in ["polygon"]):
        bbox["polygon"] = [[p[0] * reshape_ratio, p[1] * reshape_ratio] for p in bbox["polygon"]]

    image = image.resize((new_width, new_height), Image.LANCZOS)  
    new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))  
    paste_position = (0, 0)  
    new_img.paste(image, paste_position)
    
    messages = prompt_processor.generate_prompt(sample, new_width, new_height, think_mode)

    return new_img, messages, bbox


def prepare_sample_data(sample: Dict, benchmark_info: Dict, max_image_pixels: int, prompt_processor, think_mode: bool) -> tuple:
    """
    Prepares data for a single sample.
    """
    # Load image
    image_path = os.path.join(benchmark_info["image_root"], sample["images"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Process bounding box coordinates
    bbox = sample["bbox"].copy()
    
    # Resize image
    new_width, new_height = resize_image(original_width, original_height, max_image_pixels)
    
    # Further resize using smart_resize
    new_height, new_width = smart_resize(new_height, new_width, max_pixels=12845056)
    image = image.resize((new_width, new_height))
    
    # Update bounding box coordinates to the new image dimensions
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    if all(key in bbox.keys() for key in ["x1", "y1", "x2", "y2"]):
        bbox["x1"] = bbox["x1"] * scale_x
        bbox["y1"] = bbox["y1"] * scale_y
        bbox["x2"] = bbox["x2"] * scale_x
        bbox["y2"] = bbox["y2"] * scale_y
    elif all(key in bbox.keys() for key in ["polygon"]):
        bbox["polygon"] = [[p[0] * scale_x, p[1] * scale_y] for p in bbox["polygon"]]
    
    # Generate prompt
    messages = prompt_processor.generate_prompt(sample, new_width, new_height, think_mode)
    
    return image, messages, bbox


def process_response(sample: Dict, response: str, bbox: Dict, prompt_processor, think_mode: bool) -> Dict:
    """
    Processes the model's response.
    """
    # Extract coordinates
    predictions = prompt_processor.extract_coordinates(response, think_mode)
    
    # Calculate metrics
    if predictions is not None and 0 < predictions[0] < 1 and 0 < predictions[1] < 1:
        # If normalized coordinates, convert to absolute
        processed_imgsize = sample["processed_imgsize"]
        if processed_imgsize:
            predictions[0] *= processed_imgsize[0]
            predictions[1] *= processed_imgsize[1]
            
    metrics = prompt_processor.calculate_metrics(sample, predictions, bbox)
    
    # Construct result
    result = {
        "image": sample["images"],
        "instruction": sample["instruction"],
        "gt_bbox": bbox,
        "response": response,
        **metrics
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="General UI Element Localization Evaluation Framework")
    
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("--benchmark", "-b", type=str, default="screenspot-pro", help="Name of the benchmark to evaluate")
    # parser.add_argument("--prompt", type=str, default="infigui-g1", help="Name of the prompt processor to use")
    parser.add_argument("--prompt", type=str, default="star-cua", choices=list(PROMPT_PROCESSORS.keys()), help="Name of the prompt processor to use")
    parser.add_argument("--engine", type=str, default="vllm", choices=['vllm', 'hf'], help="Name of the engine to use")
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=4, help="Tensor parallelism size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Maximum number of sequences in a batch")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Whether to disable caching")
    parser.add_argument("--max-image-tokens", "-mit", type=int, default=5600, help="Maximum image tokens")
    parser.add_argument("--think-mode", type=int, default=0, help="Whether to enable thinking mode (1=enable, 0=disable)")
    parser.add_argument("--debug-mode", type=int, default=0, help="Whether to enable debug mode (1=enable, 0=disable)")
    parser.add_argument("--model-type", type=str, default='qwen2.5vl', choices=['qwen2.5vl', 'qwen2vl', 'guiactor', 'phiground'], help="Output model name, extracted from model path if not specified")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Convert flag arguments to boolean
    think_mode = bool(args.think_mode)
    debug_mode = bool(args.debug_mode)
    
    print(f"Starting evaluation - Benchmark: {args.benchmark}, Prompt: {args.prompt}")
    
    # Load benchmark information
    benchmark_info = load_benchmark_info(args.benchmark)
    print(f"Loaded benchmark: {benchmark_info['name']}")

    
    # Get prompt processor
    prompt_processor = get_prompt_processor(args.prompt)
    print(f"Using prompt processor: {args.prompt}")

    
    # Set output directory
    
    if os.path.basename(args.model_path).startswith('checkpoint'):
        model_name = os.path.basename(os.path.dirname(args.model_path)) + '_chkp' + os.path.basename(args.model_path).split('-')[-1] 
    elif os.path.basename(args.model_path) == 'huggingface':
        # e.g. /mnt/home/star-cua/star-cua/checkpoints/groundnext/vllm_7b_guiact20KLarge-5kError_from256bs5-0k_reward-01_bs64/global_step_100/actor/huggingface
        model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.model_path)))) + '_chkp' + os.path.basename(os.path.dirname(os.path.dirname(args.model_path))).split('_')[-1]
    else:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        
    
    output_dir = f"./output/{model_name}/{args.benchmark}"
    if args.engine == 'hf':
        output_dir += '_hf'
    print('Output directory:', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    max_image_pixels = args.max_image_tokens * 28 * 28 if args.max_image_tokens > 0 else 16384 * 28 * 28
    print(f"Initializing model, max image pixels: {max_image_pixels}")
    
    if args.model_type in ['qwen2.5vl', 'qwen2vl']:
        if args.engine == 'vllm':
            print('Using VLLM engine')
            from models.qwen2vl import Qwen2VL as Qwen2VL_VLLM
            llm = Qwen2VL_VLLM(
                model_path=args.model_path,
                max_model_len=max_image_pixels//28//28 + 1024,
                tensor_parallel_size=args.tensor_parallel,
                max_num_seqs=args.max_num_seqs,
                enforce_eager=True,
            )
        elif args.engine == 'hf':
            print('Using HuggingFace engine')
            from models.qwen2vl_hf import Qwen25VL as Qwen25VL_HF
            if args.model_type.startswith('qwen2.5vl'):
                print("Using Qwen2.5VL HF model")
                llm = Qwen25VL_HF(
                    model_path=args.model_path,
                    min_pixels=4*28*28,
                    max_pixels=max_image_pixels,
                    max_new_tokens=args.max_tokens
                )
            elif args.model_type.startswith('qwen2vl'):
                print("Using Qwen2VL HF model")
                pass
    elif args.model_type in ['guiactor', 'gui-actor', 'gui_actor']:
        print('Using GUI-Actor model')
        from models.gui_actor import GUI_Actor
        llm = GUI_Actor(
            model_name_or_path=args.model_path)
    elif args.model_type == 'phiground':
        print('Using PhiGround model')
        from models.phiground import PhiGround
        llm = PhiGround(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel,
            max_model_len=max_image_pixels//28//28 + 1024,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=True,
        )
    
    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_dataset(benchmark_info)
    print(f"Raw dataset size: {len(raw_dataset)}")

    
    # Standardize dataset
    dataset = []
    for sample in raw_dataset:
        standardized = standardize_sample(sample, args.benchmark)
        dataset.append(standardized)
    
    # Random sampling in debug mode
    random.seed(42)
    if debug_mode:
        dataset = random.sample(dataset, len(dataset)//50 if len(dataset) >= 10 else 1)
        print(f"Debug mode enabled, using {len(dataset)} samples.")
    
    # Prepare all data
    print("Preparing data...")
    messages_list = []
    images = []
    processed_bboxes = []


    processed_cache_dir = f"./cache/b-{benchmark_info['name']}_p-{args.prompt}_mit-{args.max_image_tokens}.pkl"
    if not args.no_cache:
        os.makedirs(os.path.dirname(processed_cache_dir), exist_ok=True)

    if os.path.exists(processed_cache_dir) and not debug_mode and not args.no_cache:
        print(f"Loading processed data cache from {processed_cache_dir}")
        with open(processed_cache_dir, "rb") as f:
            import pickle
            cache_data = pickle.load(f)
            messages_list = cache_data["messages_list"]
            images = cache_data["images"]
            processed_bboxes = cache_data["processed_bboxes"]
            dataset = cache_data["dataset"]
            print(f"Loaded {len(messages_list)} samples from cache")
    else:
        for i, sample in enumerate(tqdm(dataset, desc="Preparing samples")):
            # Prepare sample data
            if args.model_type == 'phiground':
                image, messages, bbox = prepare_sample_data_phiground(
                    sample, benchmark_info, max_image_pixels, prompt_processor, think_mode, num_crops=7
                )
            else:
                image, messages, bbox = prepare_sample_data(
                    sample, benchmark_info, max_image_pixels, prompt_processor, think_mode
                )
            messages_list.append(messages)
            images.append(image)
            processed_bboxes.append(bbox)
            
            # Update bbox information in the sample (for subsequent processing)
            dataset[i]["processed_bbox"] = bbox
            dataset[i]["processed_imgsize"] = image.size

        if not debug_mode and not args.no_cache:
            cache_data = {
                "messages_list": messages_list,
                "images": images,
                "processed_bboxes": processed_bboxes,
                "dataset": dataset}
            
            with open(processed_cache_dir, "wb") as f:
                import pickle
                pickle.dump(cache_data, f)
                print(f"Saved processed data cache to {processed_cache_dir}")
    
    # Batch generate responses
    print("Generating responses...")
    
    responses = llm.get_responses(args, messages_list=messages_list, images=images)
    
    # Process response results
    print("Processing response results...")
    results = {}
    
    for idx, (sample, response) in enumerate(tqdm(zip(dataset, responses), total=len(dataset), desc="Processing responses")):
        bbox = sample["processed_bbox"]
        # print(response)
        try:
            result = process_response(sample, response, bbox, prompt_processor, think_mode)
            results[idx] = result
        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            # Add default result using prompt processor's metric definition
            default_metrics = prompt_processor.calculate_metrics(sample, None, bbox)
            results[idx] = {
                "image": sample.get("images", ""),
                "instruction": sample.get("instruction", ""),
                "gt_bbox": bbox,
                "response": response,
                **default_metrics
            }
    
    # Calculate hierarchical statistics
    print("Calculating statistics...")
    statistics = calculate_hierarchical_statistics(results, dataset, prompt_processor)
    
    # Print statistics
    print("\n=== Evaluation Results ===")
    metric_keys = prompt_processor.get_metric_keys()
    accuracy_pairs = prompt_processor.get_accuracy_pairs()
    print_hierarchical_stats(statistics, 0, metric_keys, accuracy_pairs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reorder statistics, moving subgroups to the end
    ordered_statistics = reorder_stats_for_output(statistics)
    
    output_data = {
        "benchmark": args.benchmark,
        "prompt": args.prompt,
        "model_path": args.model_path,
        "args": vars(args),
        "statistics": ordered_statistics,
        "detailed_results": results
    }
    
    output_file = os.path.join(
        output_dir, 
        f"{timestamp}{'_t'+str(args.temperature).replace('.', '-') if args.temperature else ''}"
        f"{'_debug' if debug_mode else ''}_{args.prompt}.json"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {output_file}")
    print_dict = {}
    for k, v in ordered_statistics.items():
        if k.endswith('_accuracy'):
            print_dict[k] = v
    print(f"Overall accuracy: {print_dict}")


if __name__ == "__main__":
    main()
