import logging
import json
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetImpl:
    def __init__(self, dataset_path: str, system_prompt_path: str):
        """
        Initialize GSM8K dataset loader.
        
        Args:
            dataset_path: Path to the JSONL file
            system_prompt_path: Path to the system prompt file
        """
        self.dataset = self._load_dataset(dataset_path)
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

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
            question = example.get("question", "")
            answer = example.get("answer", "")

            # Create conversation structure
            conversation = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]

            conversations.append(conversation)
            solutions.append(answer)

        return conversations, solutions
