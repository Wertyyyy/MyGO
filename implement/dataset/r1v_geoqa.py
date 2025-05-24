import logging
from datasets import load_dataset

from data_service.typing.messages import encode_image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetImpl:
    def __init__(self, dataset_path: str, system_prompt_path: str):
        self.dataset = load_dataset(dataset_path, split="train")
        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read()

    def collate_fn(self, examples):
        conversations = []
        solutions = []

        for example in examples:
            image = example.get("image")
            img_base64 = encode_image_to_base64(image)

            conversation = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/png;base64,{img_base64}",
                        },
                        {"type": "text", "text": example.get("problem", "")},
                    ],
                },
            ]

            conversations.append(conversation)
            solutions.append(example.get("solution", example.get("ground_truth", "")))

        return conversations, solutions
