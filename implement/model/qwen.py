import logging
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_service.typing.message import Conversation
from implement.model.utils import TFBasicModelMixin, TFBasicProcessorMixin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TFModelImpl(TFBasicModelMixin):
    def __init__(self, init_params: Dict[str, Any]):
        init_params.update(
            {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            }
        )
        self.model = AutoModelForCausalLM.from_pretrained(**init_params)


class TFProcessorImpl(TFBasicProcessorMixin):
    multimodal = False

    def __init__(self, init_params: Dict[str, Any]):
        init_params.update({"padding_side": "left", "use_fast": False})
        self.processor = AutoTokenizer.from_pretrained(**init_params)
        self.prefix_ids = self.get_prefix_ids()

    def prepare_inputs(
        self, conversations: List[Conversation], max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        texts = [
            self.conversation_to_text(conversation) for conversation in conversations
        ]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        ).to("cuda")

        return inputs

    def conversation_to_token_ids(self, conversation: Conversation) -> List[int]:
        text = self.conversation_to_text(conversation)
        return self.processor(text)["input_ids"]
