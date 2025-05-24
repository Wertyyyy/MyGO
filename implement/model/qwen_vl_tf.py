import logging
import math
import torch
from typing import List, Optional

from transformers import AutoProcessor, AutoModelForVision2Seq

from data_service.typing.messages import Conversation, extract_images_from_conversation

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


class TFServerModelImpl:
    def __init__(
        self,
        model_name_or_path: str,
        min_pixels: int,
        max_pixels: int,
    ):
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side="left",
            use_fast=False,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def process_batch(
        self, conversations: List[Conversation], max_length: Optional[int] = None
    ) -> List[List[float]]:
        logger.info(f"Processing batch of {len(conversations)} conversations")
        with torch.inference_mode():
            # Process chat templates and extract images
            texts = [
                self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
                for conversation in conversations
            ]

            # Extract images from conversations using our custom function
            image_inputs = []
            for conversation in conversations:
                image_inputs.extend(extract_images_from_conversation(conversation))

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

            outputs = self.model(**inputs)

            results: List[List[float]] = []
            # Process each sequence in the batch
            for input_ids, logits in zip(inputs.input_ids, outputs.logits):
                input_ids = input_ids.cpu().tolist()
                start_index = None
                for index in range(len(input_ids)):
                    if index + 2 < len(input_ids) and list(
                        input_ids[index : index + 3]
                    ) == [151644, 77091, 198]:
                        start_index = index
                        break

                sequence_logits = logits[start_index + 2 : -1].to(torch.float32)
                logprobs_all = torch.nn.functional.log_softmax(sequence_logits, dim=-1)
                logprobs_list = []
                for index, token_id in enumerate(input_ids[start_index + 3 :]):
                    token_logprob = logprobs_all[index][token_id].item()
                    logprobs_list.append(token_logprob)

                results.append(logprobs_list)

            return results

    def get_seq_length(self, conversation: Conversation) -> int:
        if conversation[-1]["role"] == "assistant":
            pure_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        elif conversation[-1]["role"] == "user":
            pure_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

        token_ids_from_pure_text = self.processor.tokenizer(pure_text)
        length_without_image = len(token_ids_from_pure_text["input_ids"])

        image_inputs = extract_images_from_conversation(conversation)
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


class DataServerProcessorImpl:
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

    def get_seq_length(self, conversation: Conversation) -> int:
        if conversation[-1]["role"] == "assistant":
            pure_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        elif conversation[-1]["role"] == "user":
            pure_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

        token_ids_from_pure_text = self.processor.tokenizer(pure_text)
        length_without_image = len(token_ids_from_pure_text["input_ids"])

        image_inputs = extract_images_from_conversation(conversation)
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


class MainTrainingImpl:
    def __init__(
        self,
        model_name_or_path: str,
        min_pixels: int,
        max_pixels: int,
    ):
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side="left",
            use_fast=False,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def get_logprobs(
        self, conversations: List[Conversation], max_length: int
    ) -> List[torch.Tensor]:
        """
        Get log probabilities for a list of conversations.
        Returns a list of tensors containing log probabilities for each conversation.
        """

        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            for conversation in conversations
        ]

        # Extract images from conversations using our custom function (same as TFServerModelImpl)
        image_inputs = []
        for conversation in conversations:
            image_inputs.extend(extract_images_from_conversation(conversation))

        # Prepare model inputs (same pattern as TFServerModelImpl)
        inputs = self.processor(
            text=texts,
            # Important: if there are no images, we need to pass None to the processor
            images=image_inputs if len(image_inputs) > 0 else None,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        ).to("cuda")

        # Forward pass to get logits
        outputs = self.model(**inputs)

        # Compute log probabilities for each sequence
        results = []
        prefix_ids = [151644, 77091, 198]  # From main.py

        for logits, input_ids in zip(outputs.logits, inputs.input_ids):
            # Find the start index based on prefix_ids (same logic as compute_logprobs in main.py)
            start_index = None
            for index in range(len(input_ids)):
                if input_ids[index : index + 3].cpu().tolist() == prefix_ids:
                    start_index = index
                    break

            if start_index is None:
                # If prefix not found, return empty tensor
                results.append(torch.tensor([], device=logits.device))
                continue

            # Extract relevant logits (same logic as main.py)
            relevant_logits = logits[start_index + 2 : -1]
            logprobs = torch.nn.functional.log_softmax(relevant_logits, dim=-1)

            # Get token indices for response part
            token_indices = input_ids[start_index + 3 :].unsqueeze(-1)

            # Gather log probabilities for actual tokens
            logprobs_for_resp = torch.gather(logprobs, -1, token_indices).squeeze(-1)

            results.append(logprobs_for_resp)

        return results
