import logging
import torch
from typing import List, Optional, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

from data_service.typing.message import Conversation
from implement.model.basic import TFBasicModelMixin, TFBasicProcessorMixin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TFModelImpl(TFBasicModelMixin):
    multimodal = False

    def __init__(self, model_name_or_path: str):
        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )


class TFProcessorImpl(TFBasicProcessorMixin):
    multimodal = False

    def __init__(self, model_name_or_path: str):
        self.processor = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            use_fast=False,
        )
        self.prefix_ids = self.get_prefix_ids()

    def get_seq_length(self, conversation: Conversation) -> int:
        if conversation.get_last_role() == "assistant":
            text = self.processor.apply_chat_template(
                conversation.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        elif conversation.get_last_role() == "user":
            text = self.processor.apply_chat_template(
                conversation.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )

        token_ids = self.processor.tokenize(text)
        return len(token_ids)

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

        # Prepare model inputs
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        ).to("cuda")

        return inputs
