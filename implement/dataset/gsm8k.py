import logging
import json
from typing import Optional

from datasets import Dataset

from data_service.typing.message import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetImpl:
    def __init__(self, dataset_path: str, system_prompt_path: Optional[str] = None, prompt_template_path: Optional[str] = None):
        """
        Initialize GSM8K dataset loader.

        Args:
            dataset_path: Path to the JSONL file
            system_prompt_path: Path to the system prompt file
        """
        self.dataset = self._load_dataset(dataset_path)
        if system_prompt_path is not None:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
                logger.info(f"Loaded system prompt {repr(self.system_prompt)}")
        else:
            self.system_prompt = None
        if prompt_template_path is not None:
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
                logger.info(f"Loaded prompt template {repr(self.prompt_template)}")
        else:
            self.prompt_template = None

    def _load_dataset(self, dataset_path: str):
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
        """
        Collate function to process a batch of examples.

        Args:
            examples: List of examples from the dataset

        Returns:
            conversations: List of conversation structures
            solutions: List of answer strings
        """
        conversations = []
        solutions = []

        for example in examples:
            question = example.get("question")
            answer = example.get("answer").split("####")[1].strip()

            if self.prompt_template is not None:
                prompt = self.prompt_template.format(instruction=question)
            else:
                prompt = question

            # Create conversation structure
            if self.system_prompt is not None:
                conversation = Conversation(
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
            else:
                conversation = Conversation(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )

            conversations.append(conversation)
            solutions.append(answer)

        return conversations, solutions
