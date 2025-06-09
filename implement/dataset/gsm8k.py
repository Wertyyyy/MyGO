import logging
import json
from typing import Optional

from datasets import Dataset

from data_service.typing.message import Conversation
from implement.dataset.utils import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDatasetImpl:
    def __init__(
        self,
        dataset_path: str,
        system_prompt_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        self.dataset = self._load_dataset(dataset_path)

        self.system_prompt = (
            load_prompt(system_prompt_path) if system_prompt_path is not None else None
        )
        self.prompt_template = (
            load_prompt(template_path) if template_path is not None else None
        )

    @staticmethod
    def _load_dataset(dataset_path: str):
        """Load dataset from local JSONL file"""
        logger.info(f"Loading local JSONL file: {dataset_path}")
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    def collate_fn(self, examples):
        conversations = []
        solutions = []

        for example in examples:
            question = example.get("question")
            answer = example.get("answer").split("####")[1].strip()

            if self.prompt_template is not None:
                prompt = self.prompt_template.format(instruction=question)
            else:
                prompt = question

            message_list = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            if self.system_prompt is not None:
                message_list.insert(
                    0,
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                )

            conversation = Conversation(messages=message_list)
            conversations.append(conversation)
            solutions.append(answer)

        return conversations, solutions


TestDatasetImpl = TrainingDatasetImpl
