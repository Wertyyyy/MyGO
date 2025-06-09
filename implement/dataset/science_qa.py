import logging
from datasets import load_dataset
import string
from typing import List, Optional

from datasets import concatenate_datasets, Dataset

from data_service.typing.message import encode_image_to_base64, Conversation
from implement.dataset.utils import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetImpl:
    def __init__(
        self,
        dataset: Dataset,
        system_prompt_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        original_count = len(dataset)

        filtered_dataset = dataset.filter(
            lambda example: example.get("image") is not None
        )
        filtered_count = len(filtered_dataset)

        logger.info(f"Before filtering: {original_count}")
        logger.info(f"After filtering: {filtered_count}")
        logger.info(f"Filtered out: {original_count - filtered_count}")

        self.dataset = filtered_dataset
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
            # Get question and choices
            question = example.get("question", "")
            choices = example.get("choices", [])
            answer_idx = example.get("answer", 0)

            # Format choices as ABCD options
            choice_labels = list(string.ascii_uppercase)
            formatted_choices = []
            for i, choice in enumerate(choices):
                if i < len(choice_labels):
                    formatted_choices.append(f"{choice_labels[i]}. {choice}")

            # Combine question with choices
            choices_text = "\n".join(formatted_choices)
            if question.strip():  # Only add question if it's not empty
                full_question = (
                    f"{question}\n\n{choices_text}" if choices_text else question
                )
            else:
                full_question = (
                    choices_text
                    if choices_text
                    else "Please select the correct answer."
                )

            # Apply prompt template if available
            if self.prompt_template is not None:
                prompt = self.prompt_template.format(instruction=full_question)
            else:
                prompt = full_question

            # Convert answer index to letter
            answer_letter = (
                choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"
            )

            # Prepare user content - handle cases with/without images
            user_content = []

            # Add image if present
            image = example.get("image")
            img_base64 = encode_image_to_base64(image)
            user_content.append(
                {
                    "type": "image",
                    "image": img_base64,
                }
            )

            # Add text content
            user_content.append({"type": "text", "text": prompt})

            message_list = [
                {
                    "role": "user",
                    "content": user_content,
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
            solutions.append(answer_letter)

        return conversations, solutions


class TrainingDatasetImpl(DatasetImpl):
    def __init__(
        self,
        dataset_path: str,
        system_prompt_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        train_dataset = load_dataset(dataset_path, split="train")
        val_dataset = load_dataset(dataset_path, split="validation")
        original_dataset = concatenate_datasets([train_dataset, val_dataset])
        super().__init__(original_dataset, system_prompt_path, template_path)


class TestDatasetImpl(DatasetImpl):
    def __init__(
        self,
        dataset_path: str,
        system_prompt_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        test_dataset = load_dataset(dataset_path, split="test")
        super().__init__(test_dataset, system_prompt_path, template_path)
