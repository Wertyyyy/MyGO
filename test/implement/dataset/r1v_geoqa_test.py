import logging

from implement.dataset.r1v_geoqa import DatasetImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "/aiarena/project/2014-p15/datasets/GeoQA"
SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

def test_collate_function(dataset):
    logger.info("Testing collate function...")

    try:
        test_sizes = [1, 3, 5] if len(dataset.dataset) >= 5 else [1, min(3, len(dataset.dataset))]
        
        for size in test_sizes:
            logger.info(f"Testing collate with {size} sample(s)...")
            
            # Extract samples from dataset
            examples = [dataset.dataset[i] for i in range(size)]
            
            # Test collate function
            messages, solutions = dataset.collate_fn(examples)
            
            # Verify results
            assert len(messages) == size, f"Expected {size} messages, got {len(messages)}"
            assert len(solutions) == size, f"Expected {size} solutions, got {len(solutions)}"
            
            # Check structure of first message
            if size > 0:
                first_message = messages[0]
                assert len(first_message) == 2, "Message should have 2 parts (system + user)"
                assert first_message[0]["role"] == "system", "First part should be system message"
                assert first_message[0]["content"] == dataset.system_prompt, "System message content mismatch"
                
                assert first_message[1]["role"] == "user", "Second part should be user message"
                user_content = first_message[1]["content"]
                assert len(user_content) == 2, "User content should have image + text"
                assert user_content[0]["type"] == "image", "First content should be image"
                assert user_content[1]["type"] == "text", "Second content should be text"
                
                # Check image encoding
                image_data = user_content[0]["image"]
                assert image_data.startswith("data:image/"), "Image should be base64 encoded"
                assert "base64," in image_data, "Image should contain base64 data"
                
                logger.info("  ✓ Sample message structure valid")
                logger.info(f"    Problem text: {user_content[1]['text'][:100]}...")
                logger.info(f"    Solution: {solutions[0][:100]}...")
                logger.info(f"    Image data size: {len(image_data)} characters")
            
            logger.info(f"  ✓ Collate test passed for {size} samples")
            
    except Exception as e:
        logger.error(f"Collate function test failed: {str(e)}")

def main():
    logger.info("Starting r1v_geoqa DatasetImpl tests...")
    
    dataset = DatasetImpl(DATASET_PATH, SYSTEM_PROMPT)
    test_collate_function(dataset)

if __name__ == "__main__":
    main() 