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
            # Get question and choices
            question = example.get("question", "")
            choices = example.get("choices", [])
            answer_idx = example.get("answer", 0)
            
            # Format choices as ABCD options
            choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            formatted_choices = []
            for i, choice in enumerate(choices):
                if i < len(choice_labels):
                    formatted_choices.append(f"{choice_labels[i]}. {choice}")
            
            # Combine question with choices
            choices_text = "\n".join(formatted_choices)
            if question.strip():  # Only add question if it's not empty
                full_question = f"{question}\n\n{choices_text}" if choices_text else question
            else:
                full_question = choices_text if choices_text else "Please select the correct answer."
            
            # Convert answer index to letter
            answer_letter = choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"

            # Prepare user content - handle cases with/without images
            user_content = []
            
            # Add image if present
            image = example.get("image")
            if image is not None:
                try:
                    img_base64 = encode_image_to_base64(image)
                    user_content.append({
                        "type": "image",
                        "image": f"data:image/png;base64,{img_base64}",
                    })
                except Exception as e:
                    logger.warning(f"Failed to encode image: {str(e)}")
            
            # Add text content if there's any meaningful text
            if full_question.strip():
                user_content.append({"type": "text", "text": full_question})
            
            # Ensure we have at least some content
            if not user_content:
                user_content.append({"type": "text", "text": "Please select the correct answer from the given choices."})

            conversation = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

            conversations.append(conversation)
            solutions.append(answer_letter)

        return conversations, solutions
