import logging
import json
import os
from datasets import load_dataset, Dataset
from typing import List, Optional
from PIL import Image

from data_service.typing.message import Conversation, encode_image_to_base64
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
        self.dataset = load_dataset(dataset_path, split="train")
        self.system_prompt = (
            load_prompt(system_prompt_path) if system_prompt_path is not None else None
        )
        self.prompt_template = (
            load_prompt(template_path) if template_path is not None else None
        )

    def collate_fn(self, examples) -> List[Conversation]:
        conversations = []
        solutions = []

        for example in examples:
            if self.prompt_template is not None:
                prompt = self.prompt_template.format(instruction=example.get("problem"))
            else:
                prompt = example.get("problem")

            message_list = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example.get("image")},
                        {"type": "text", "text": prompt},
                    ],
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
            solutions.append(example.get("solution", example.get("ground_truth")))

        return conversations, solutions


class TestDatasetImpl:
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
        self.dataset_path = dataset_path

    @staticmethod
    def _load_dataset(dataset_path: str):
        """Load dataset from local JSONL file and filter by image size"""
        logger.info(f"Loading local JSONL file: {dataset_path}")
        data = []
        filtered_count = 0
        total_count = 0

        with open(
            os.path.join(dataset_path, "geoqa_test_prompt.jsonl"), "r", encoding="utf-8"
        ) as f:
            for line in f:
                line = line.strip()
                if line:
                    total_count += 1
                    item = json.loads(line)
                    image_path = os.path.join(dataset_path, item.get("image_path"))

                    with Image.open(image_path) as img:
                        width, height = img.size
                        min_dimension = min(width, height)

                        if min_dimension >= 28:
                            data.append(item)
                        else:
                            filtered_count += 1
                            logger.debug(
                                f"Filtered out image {image_path} with size {width}x{height} (min: {min_dimension})"
                            )

        logger.info(f"Loaded {len(data)} samples from {total_count} total samples")
        logger.info(f"Filtered out {filtered_count} samples due to image size < 28px")

        return Dataset.from_list(data)

    def collate_fn(self, examples) -> List[Conversation]:
        conversations = []
        solutions = []

        for example in examples:
            question = example.get("question")
            image_path = example.get("image_path")
            ground_truth = example.get("ground_truth")

            if self.prompt_template is not None:
                prompt = self.prompt_template.format(instruction=question)
            else:
                prompt = question

            message_list = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": encode_image_to_base64(
                                os.path.join(self.dataset_path, image_path)
                            ),
                        },
                        {"type": "text", "text": prompt},
                    ],
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
            solutions.append(ground_truth)

        return conversations, solutions
