import logging
from datasets import load_dataset
from typing import List

from data_service.typing.message import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetImpl:
    def __init__(self, dataset_path: str, system_prompt_path: str):
        self.dataset = load_dataset(dataset_path, split="train")
        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read()

    def collate_fn(self, examples) -> List[Conversation]:
        conversations = []
        solutions = []

        for example in examples:
            conversation = Conversation(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": example.get("image")},
                            {"type": "text", "text": example.get("problem")},
                        ],
                    },
                ]
            )

            conversations.append(conversation)
            solutions.append(example.get("solution", example.get("ground_truth")))

        return conversations, solutions
