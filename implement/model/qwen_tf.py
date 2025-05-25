import logging
import torch
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from data_service.typing.messages import Conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TFServerModelImpl:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
    ):
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            use_fast=False,
        )
        # Set pad token if not exists
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
        self.max_length = max_length

    def process_batch(
        self, conversations: List[Conversation], max_length: Optional[int] = None
    ) -> List[List[float]]:
        logger.info(f"Processing batch of {len(conversations)} conversations")
        with torch.inference_mode():
            # Process chat templates
            texts = [
                self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
                for conversation in conversations
            ]

            # Prepare model inputs
            inputs = self.processor(
                texts,
                return_tensors="pt",
                max_length=max_length or self.max_length,
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

                if start_index is None:
                    results.append([])
                    continue

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

        token_ids = self.processor(pure_text, return_tensors="pt")
        return len(token_ids["input_ids"][0])


class DataServerProcessorImpl:
    def __init__(self, model_name_or_path: str):
        self.processor = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            use_fast=False,
        )
        # Set pad token if not exists
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

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

        token_ids = self.processor(pure_text, return_tensors="pt")
        return len(token_ids["input_ids"][0])


class MainTrainingImpl:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
    ):
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            use_fast=False,
        )
        # Set pad token if not exists
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
        self.max_length = max_length

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

        # Prepare model inputs
        inputs = self.processor(
            texts,
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
