from typing import List, Union, Dict, Any, Optional
from openai import responses
from tqdm import tqdm
from transformers import AutoConfig
from PIL import Image
from vllm import LLM, SamplingParams

class PhiGround:
    def __init__(
        self,
        model_path: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.99,
        enforce_eager: bool = False,
        max_model_len: int = 8192,
        limit_mm_per_prompt: Dict[str, int] = {"image": 1, "video": 0},
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
    ):
        """
        Initialize Qwen2VL/Qwen2.5VL inference class
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of tensor parallel processes
            gpu_memory_utilization: GPU memory utilization ratio
            enforce_eager: Whether to enforce eager execution
            max_model_len: Maximum sequence length
            limit_mm_per_prompt: Limit multimodal inputs per prompt
            min_pixels: Minimum number of pixels for image processing
            max_pixels: Maximum number of pixels for image processing
            max_num_seqs: Maximum number of sequences
        """
        kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
            "limit_mm_per_prompt": limit_mm_per_prompt,
            "trust_remote_code": True
        }
        
        if max_num_seqs:
            kwargs["max_num_seqs"] = max_num_seqs
            
        if min_pixels or max_pixels:
            kwargs["mm_processor_kwargs"] = {
                "min_pixels": min_pixels if min_pixels else 4 * 28 * 28,
                "max_pixels": max_pixels if max_pixels else 16384 * 28 * 28,
            }


        self.llm = LLM(**kwargs)
        self.system_prompt = "You are a helpful assistant."
        self.debug_mode = False
        
        
    def _format_prompt(self, messages: List[Dict[str, Any]], images: List[Image.Image]) -> str:
        """
        Format conversation history and handle custom image positions
        
        Args:
            messages: Conversation history
            images: List of images
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        assert messages[0]["role"] != "system"
        instruction = messages[0]["content"]
        
        ctns = [{"type": "text", "text": instruction}]
        ctns.append({"type": "image", "path": "<image>"})
  
        img_id = 1  
        prompt = "<|user|> \n"  
        for ct in ctns:  
            if ct["type"] == "text":  
                prompt += ct["text"]  
            else:  
                prompt += f"<|image_{img_id}|> \n"  
                img_id += 1  

        prompt += "<|end|> \n<|assistant|>"
        
        if self.debug_mode:
            print("### FORMATTED PROMPT:")
            print(prompt)

        return prompt

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Chat interface supporting single or batch conversations
        
        Args:
            messages: Single conversation history or list of conversation histories
            images: Single image, list of images, or batch list of images
            temperature: Sampling temperature
            max_tokens: Maximum generation length
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated response or list of responses
        """
        # Check if this is a batch request
        is_batch = isinstance(messages[0], list)
        if not is_batch:
            messages = [messages]
            images = [images] if isinstance(images, Image.Image) else ([images] if images is not None else [None])
            
        # Prepare inputs
        inputs = []
        assert len(messages) == len(images)
        for msg, img_list in zip(messages, images):
            if img_list is not None:
                img_list = [img_list] if isinstance(img_list, Image.Image) else img_list
                prompt = self._format_prompt(msg, img_list)
                
                # Build multimodal data dictionary
                input_data = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img_list
                    }
                }
            else:
                prompt = self._format_prompt(msg, [])
                input_data = {
                    "prompt": prompt,
                    "multi_modal_data": {}
                }
                
            inputs.append(input_data)
            
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Execute inference
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        if self.debug_mode:
            print("### RESPONSES:")
            print(responses)
            print("### END OF RESPONSES")
            print("="*50)
        
        return responses[0] if not is_batch else responses

    def get_responses(self, args, messages_list, images):
        responses = []
        if args.debug_mode == 1:
            self.debug_mode = True
        total_samples = len(messages_list)
        progress_bar = tqdm(total=total_samples, desc="Generating responses")
        
        for i in range(0, len(messages_list), args.batch_size):
            batch_messages = messages_list[i:i+args.batch_size] 
            batch_images = images[i:i+args.batch_size]
            
            # Generate responses
            batch_responses = self.chat(
                batch_messages, 
                batch_images, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature
            )
            responses.extend(batch_responses)
            # print([responses[-1]])
            
            progress_bar.update(len(batch_messages))
        
        progress_bar.close()
    
        return responses
