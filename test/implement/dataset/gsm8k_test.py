import logging
import os

from implement.dataset.gsm8k import DatasetImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "/aiarena/project/2014-p15/datasets/gsm8k_train.json"
SYSTEM_PROMPT_PATH = "/aiarena/project/2014-p15/MyGO_v1/data_service/prompt/math_r1v.txt"


def test_dataset_loading():
    """Test basic dataset loading functionality"""
    logger.info("Testing dataset loading...")
    
    # Check if dataset file exists
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Dataset file not found: {DATASET_PATH}")
        logger.info("Skipping dataset loading test")
        return
    
    # Check if system prompt file exists
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        logger.warning(f"System prompt file not found: {SYSTEM_PROMPT_PATH}")
        logger.info("Skipping dataset loading test")
        return
    
    # Initialize dataset
    dataset = DatasetImpl(DATASET_PATH, SYSTEM_PROMPT_PATH)
    
    # Test basic properties
    logger.info(f"Dataset size: {len(dataset.dataset)}")
    assert len(dataset.dataset) > 0, f"Dataset should not be empty"
    
    # Test dataset indexing
    first_item = dataset.dataset[0]
    logger.info(f"First item keys: {first_item.keys()}")
    assert "question" in first_item
    assert "answer" in first_item
    
    logger.info("✓ Dataset loading test passed")


def test_collate_function():
    """Test the collate function with sample data"""
    logger.info("Testing collate function...")
    
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Dataset file not found: {DATASET_PATH}")
        logger.info("Skipping collate function test")
        return
    
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        logger.warning(f"System prompt file not found: {SYSTEM_PROMPT_PATH}")
        logger.info("Skipping collate function test")
        return
    
    dataset = DatasetImpl(DATASET_PATH, SYSTEM_PROMPT_PATH)
    
    # Test single example
    example = dataset.dataset[0]
    conversations, solutions = dataset.collate_fn([example])
    
    logger.info(f"Number of conversations: {len(conversations)}")
    logger.info(f"Number of solutions: {len(solutions)}")
    
    assert len(conversations) == 1
    assert len(solutions) == 1
    
    conversation = conversations[0]
    solution = solutions[0]
    
    # Verify conversation structure
    assert len(conversation) == 2  # system + user
    assert conversation[0]["role"] == "system"
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"] == example["question"]
    assert solution == example["answer"]
    
    logger.info("✓ Single example collate test passed")
    
    # Test batch processing (use first 3 examples or less if dataset is smaller)
    batch_size = min(3, len(dataset.dataset))
    examples = [dataset.dataset[i] for i in range(batch_size)]
    conversations, solutions = dataset.collate_fn(examples)
    
    assert len(conversations) == len(examples)
    assert len(solutions) == len(examples)
    
    logger.info("✓ Batch collate test passed")


def main():
    """Run all tests"""
    logger.info("Starting GSM8K dataset tests...")
    
    try:
        test_dataset_loading()
        test_collate_function()
        
        logger.info("🎉 All GSM8K dataset tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 