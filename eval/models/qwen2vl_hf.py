from typing import List, Union, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers.generation import GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig

from PIL import Image

from qwen_vl_utils import process_vision_info


class Qwen25VL:
    def __init__(
        self,
        model_path: str = "",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Initialize Qwen2VL/Qwen2.5VL inference class
        
        Args:
            model_path: Path to the model
            min_pixels: Minimum number of pixels for image processing
            max_pixels: Maximum number of pixels for image processing
            max_new_tokens: Maximum number of generation length
        """
        
        self.max_pixels = max_pixels if max_pixels else 16384 * 28 * 28
        self.min_pixels = min_pixels if min_pixels else 4 * 28 * 28
        self.max_new_tokens = max_new_tokens if max_new_tokens else 2048
        self.debug_mode = False
            
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        if "text_config" in config:
            # Remove text_config to avoid conflicts
            print('removing text_config from model config')
            delattr(config, "text_config") 
            config.save_pretrained(model_path)
        

        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.model.generation_config.temperature = 0.0
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.model.generation_config.do_sample = False
        self.model.generation_config.use_cache = True
        
        self.system_prompt = "You are a helpful assistant."
        
        
        
    def _format_prompt(self, messages: List[Dict[str, Any]], images=[]) -> str:
        """
        Format conversation history and handle custom image positions
        
        Args:
            messages: Conversation history
            images: List of images
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        new_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            cnts = content.split('<image>')
            num_imgs = len(cnts) - 1
            if num_imgs == 0:
                new_messages.append({"role": role, "content": [{"type": "text", "text": cnts[0]}]})
                continue
            else:
                new_content = []
                for idx, cnt in enumerate(cnts):
                    if cnt.strip() != "":
                        new_content.append({"type": "text", "text": cnt.strip()})

                    if idx < num_imgs:
                        new_content.append({"type": "image", "image": images[idx]})

                new_messages.append({"role": role, "content": new_content})
        if self.debug_mode:
            print("Formatted Messages with Images:")
            print(new_messages) 
        # Verify all images are used
        
        prompt = self.processor.apply_chat_template(
            new_messages, tokenize=False, add_generation_prompt=True
        )
        
        return prompt

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        
        
        prompt = self._format_prompt(messages, images)
        
        if self.debug_mode:
            print("Formatted Prompt:")
            print(prompt)
            print("End of Prompt")
            print("="*50)
        
        # print(prompt)
        
        inputs = self.processor(
            text=[prompt],
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        if self.debug_mode:
            print("Response:")
            print(response)
            print("End of Response")
            print("="*50)
        
        return response


    def get_responses(self, args, messages_list, images):
        res = []
        self.debug_mode = args.debug_mode == 1
        for messages, image in tqdm(zip(messages_list, images), desc="Generating responses"):
            # Process each instance and generate a response
            response = self.chat(messages, [image])
            res.append(response)
        return res