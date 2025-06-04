from typing import Dict, Any, List
import logging

import torch

from data_service.typing.message import Conversation, Message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TFBasicModelMixin:
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self.model(**inputs)
        return outputs


class TFBasicProcessorMixin:
    def get_prefix_ids(self) -> List[int]:
        conversation = Conversation(
            messages=[Message(role="user", content="Hello, how are you?")]
        )
        text_without_prefix = self.processor.apply_chat_template(
            conversation.model_dump()["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text_with_prefix = self.processor.apply_chat_template(
            conversation.model_dump()["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix = text_with_prefix.replace(text_without_prefix, "")
        if self.multimodal:
            return self.processor.tokenizer(prefix)["input_ids"]
        else:
            return self.processor(prefix)["input_ids"]

    def get_batched_resp_logits_and_input_ids(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get batched logits from outputs."""
        prefix_len = len(self.get_prefix_ids())
        prefix_ids = self.get_prefix_ids()

        # Compute log probabilities for each sequence
        batched_resp_logits = []
        batched_input_ids = []
        for logits, input_ids in zip(outputs["logits"], inputs["input_ids"]):
            # Find the start index based on prefix_ids from the end (for multi-turn conversations)
            start_index = None
            for index in range(len(input_ids) - prefix_len, -1, -1):
                if input_ids[index : index + prefix_len].cpu().tolist() == prefix_ids:
                    start_index = index
                    break

            if start_index is None:
                # If prefix not found, return empty tensor
                batched_resp_logits.append(torch.tensor([], device=logits.device))
                batched_input_ids.append(torch.tensor([], device=input_ids.device))
                logger.warning(f"Prefix not found in input_ids: {input_ids}")
                continue

            # Extract relevant logits
            relevant_logits = logits[start_index + prefix_len - 1 : -1]
            relevant_input_ids = input_ids[start_index + prefix_len :]
            batched_resp_logits.append(relevant_logits)
            batched_input_ids.append(relevant_input_ids)

        return batched_resp_logits, batched_input_ids

    def get_batched_entropy(
        self, batched_logits: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Get batched entropy from logits."""

        results = []
        for logits in batched_logits:
            if logits.numel() == 0:  # Handle empty tensors
                results.append(torch.tensor([], device=logits.device))
            else:
                pd = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(
                    pd * logits, dim=-1
                )
                results.append(entropy)

        return results

    def get_batched_logprobs(
        self, batched_logits: List[torch.Tensor], batched_input_ids: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Get batched log probabilities from logits."""
        results = []
        for logits, input_ids in zip(batched_logits, batched_input_ids):
            if logits.numel() == 0:  # Handle empty tensors
                results.append(torch.tensor([], device=logits.device))
            else:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_indices = input_ids.unsqueeze(-1)

                # Gather log probabilities for actual tokens
                logprobs_for_resp = torch.gather(logprobs, -1, token_indices).squeeze(
                    -1
                )

                results.append(logprobs_for_resp)

        return results
