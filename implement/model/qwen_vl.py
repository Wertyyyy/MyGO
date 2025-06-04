import logging
import math
import torch
from typing import List, Optional, Dict, Any

from transformers import AutoProcessor, AutoModelForVision2Seq

from data_service.typing.message import Conversation
from implement.model.basic import TFBasicModelMixin, TFBasicProcessorMixin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
    max_ratio: int = 200,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class TFModelImpl(TFBasicModelMixin):
    multimodal = True

    def __init__(self, model_name_or_path: str):
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )


class TFProcessorImpl(TFBasicProcessorMixin):
    multimodal = True

    def __init__(self, model_name_or_path: str, min_pixels: int, max_pixels: int):
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side="left",
            use_fast=False,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.prefix_ids = self.get_prefix_ids()

    def get_seq_length(self, conversation: Conversation) -> int:
        if conversation.get_last_role() == "assistant":
            pure_text = self.processor.apply_chat_template(
                conversation.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        elif conversation.get_last_role() == "user":
            pure_text = self.processor.apply_chat_template(
                conversation.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )

        token_ids_from_pure_text = self.processor.tokenizer(pure_text)
        length_without_image = len(token_ids_from_pure_text["input_ids"])

        image_inputs = conversation.get_images()
        image_token_count = 0

        for image in image_inputs:
            image_height, image_width = image.size
            resized_height, resized_width = smart_resize(
                image_height, image_width, self.min_pixels, self.max_pixels
            )
            # The number of tokens per image patch = (height * width) / (patch_size^2)
            # Qwen2-VL uses 28x28 patches
            image_token_count += int(resized_height * resized_width / (28 * 28))

        total_length = length_without_image + image_token_count - len(image_inputs)

        return total_length

    def prepare_inputs(
        self, conversations: List[Conversation], max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        texts = [
            self.processor.apply_chat_template(
                conversation.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            for conversation in conversations
        ]

        # Extract images from conversations
        image_inputs = []
        for conversation in conversations:
            image_inputs.extend(conversation.get_images())

        # Prepare model inputs
        inputs = self.processor(
            text=texts,
            # Important: if there are no images, we need to pass None to the processor
            images=image_inputs if len(image_inputs) > 0 else None,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        ).to("cuda")

        return inputs
