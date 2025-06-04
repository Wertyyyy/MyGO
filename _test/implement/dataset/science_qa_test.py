import logging
import os

from implement.dataset.science_qa import DatasetImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "/aiarena/project/2014-p15/datasets/ScienceQA"
SYSTEM_PROMPT_PATH = "/aiarena/project/2014-p15/MyGO_v1/data_service/prompt/science_qa.txt"


def test_dataset_loading():
    """Test basic dataset loading functionality"""
    logger.info("Testing dataset loading...")
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Dataset path not found: {DATASET_PATH}")
        return None
    
    # Check if system prompt file exists
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        logger.warning(f"System prompt file not found: {SYSTEM_PROMPT_PATH}")
        # Create a simple prompt file for testing
        os.makedirs(os.path.dirname(SYSTEM_PROMPT_PATH), exist_ok=True)
        with open(SYSTEM_PROMPT_PATH, "w") as f:
            f.write("You are a helpful AI assistant. Answer the following multiple choice question by selecting the correct option (A, B, C, or D).")
        logger.info(f"Created system prompt file: {SYSTEM_PROMPT_PATH}")
    
    try:
        # Initialize dataset
        dataset = DatasetImpl(DATASET_PATH, SYSTEM_PROMPT_PATH)
        
        # Test basic properties
        logger.info(f"Dataset size: {len(dataset.dataset)}")
        assert len(dataset.dataset) > 0, "Dataset should not be empty"
        
        # Test dataset indexing
        first_item = dataset.dataset[0]
        logger.info(f"First item keys: {first_item.keys()}")
        
        # Check required fields for ScienceQA
        required_fields = ["question", "choices", "answer"]
        for field in required_fields:
            if field not in first_item:
                logger.warning(f"Missing field: {field}")
        
        # Log sample data structure
        logger.info(f"Sample question: {first_item.get('question', 'N/A')[:100]}...")
        logger.info(f"Sample choices: {first_item.get('choices', [])}")
        logger.info(f"Sample answer index: {first_item.get('answer', 'N/A')}")
        logger.info(f"Has image: {'image' in first_item and first_item['image'] is not None}")
        
        logger.info("✓ Dataset loading test passed")
        return dataset
        
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return None


def test_dataset_statistics(dataset):
    """Analyze dataset statistics"""
    if dataset is None:
        logger.info("Skipping statistics test - dataset not available")
        return
    
    logger.info("Analyzing dataset statistics...")
    
    total_samples = len(dataset.dataset)
    logger.info(f"Total dataset samples: {total_samples}")
    
    # Count samples with/without images and text
    stats = {
        "with_image": 0,
        "without_image": 0,
        "with_question": 0,
        "without_question": 0,
        "with_choices": 0,
        "without_choices": 0,
        "choice_counts": {}
    }
    
    # Sample a subset for analysis (or all if dataset is small)
    analysis_size = min(1000, total_samples)
    logger.info(f"Analyzing first {analysis_size} samples...")
    
    for i in range(analysis_size):
        example = dataset.dataset[i]
        
        # Count image presence
        if example.get("image") is not None:
            stats["with_image"] += 1
        else:
            stats["without_image"] += 1
        
        # Count question presence
        question = example.get("question", "")
        if question and question.strip():
            stats["with_question"] += 1
        else:
            stats["without_question"] += 1
        
        # Count choices presence and distribution
        choices = example.get("choices", [])
        if choices and len(choices) > 0:
            stats["with_choices"] += 1
            choice_count = len(choices)
            stats["choice_counts"][choice_count] = stats["choice_counts"].get(choice_count, 0) + 1
        else:
            stats["without_choices"] += 1
    
    # Log statistics
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        if key != "choice_counts":
            percentage = (value / analysis_size) * 100
            logger.info(f"  {key}: {value} ({percentage:.1f}%)")
    
    logger.info("Choice count distribution:")
    for count, freq in sorted(stats["choice_counts"].items()):
        percentage = (freq / analysis_size) * 100
        logger.info(f"  {count} choices: {freq} samples ({percentage:.1f}%)")


def test_collate_function(dataset):
    """Test collate function with real dataset"""
    if dataset is None:
        logger.info("Skipping collate function test - dataset not available")
        return
    
    logger.info("Testing collate function...")
    
    try:
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
        
        # Check user content
        user_content = conversation[1]["content"]
        logger.info(f"User content types: {[item.get('type') for item in user_content]}")
        
        # Verify content structure
        has_image = any(item.get("type") == "image" for item in user_content)
        has_text = any(item.get("type") == "text" for item in user_content)
        
        assert has_image or has_text, "Should have either image or text content"
        
        if has_image:
            image_content = next(item for item in user_content if item.get("type") == "image")
            image_data = image_content["image"]
            assert image_data.startswith("data:image/"), "Image should be base64 encoded"
            assert "base64," in image_data, "Image should contain base64 data"
            logger.info("✓ Image encoding verified")
        
        if has_text:
            text_content = next(item for item in user_content if item.get("type") == "text")
            logger.info(f"Sample text: {text_content['text'][:200]}...")
            
            # Check for ABCD formatting if there are choices
            if example.get("choices"):
                text = text_content['text']
                choices = example.get("choices", [])
                for i, choice in enumerate(choices[:4]):  # Check first 4 choices
                    expected_label = chr(ord('A') + i)
                    assert f"{expected_label}. " in text, f"Should contain choice {expected_label}"
                logger.info("✓ ABCD formatting verified")
        
        # Check if solution is a valid letter
        assert solution in ["A", "B", "C", "D"], f"Solution should be A, B, C, or D, got: {solution}"
        logger.info(f"Answer: {solution}")
        
        logger.info("✓ Single example test passed")
        
        # Test batch processing
        batch_size = min(10, len(dataset.dataset))
        examples = [dataset.dataset[i] for i in range(batch_size)]
        conversations, solutions = dataset.collate_fn(examples)
        
        assert len(conversations) == len(examples)
        assert len(solutions) == len(examples)
        
        # Check all solutions are valid letters
        for i, sol in enumerate(solutions):
            assert sol in ["A", "B", "C", "D"], f"All solutions should be letters, got: {sol} at index {i}"
        
        # Log batch statistics
        batch_stats = {
            "with_image": sum(1 for conv in conversations if any(item.get("type") == "image" for item in conv[1]["content"])),
            "with_text": sum(1 for conv in conversations if any(item.get("type") == "text" for item in conv[1]["content"]))
        }
        logger.info(f"Batch processing stats: {batch_stats}")
        
        logger.info("✓ Batch processing test passed")
        
    except Exception as e:
        logger.error(f"Collate function test failed: {str(e)}")
        raise


def test_edge_cases(dataset):
    """Test with diverse examples from the dataset"""
    if dataset is None:
        logger.info("Skipping edge cases test - dataset not available")
        return
    
    logger.info("Testing diverse examples...")
    
    try:
        # Test with more examples to find edge cases
        test_indices = list(range(min(50, len(dataset.dataset))))
        
        no_image_count = 0
        no_text_count = 0
        various_choice_counts = {}
        
        for i in test_indices:
            example = dataset.dataset[i]
            conversations, solutions = dataset.collate_fn([example])
            
            # Should always produce valid output
            assert len(conversations) == 1
            assert len(solutions) == 1
            assert solutions[0] in ["A", "B", "C", "D"]
            
            # Check content types
            user_content = conversations[0][1]["content"]
            has_image = any(item.get("type") == "image" for item in user_content)
            has_text = any(item.get("type") == "text" for item in user_content)
            
            if not has_image:
                no_image_count += 1
            if not has_text:
                no_text_count += 1
            
            # Track choice counts
            choices = example.get("choices", [])
            choice_count = len(choices)
            various_choice_counts[choice_count] = various_choice_counts.get(choice_count, 0) + 1
        
        logger.info(f"Tested {len(test_indices)} examples:")
        logger.info(f"  Examples without images: {no_image_count}")
        logger.info(f"  Examples without text: {no_text_count}")
        logger.info(f"  Choice count distribution: {various_choice_counts}")
        
        logger.info("✓ Edge cases test completed")
        
    except Exception as e:
        logger.error(f"Edge cases test failed: {str(e)}")
        raise


def main():
    """Run all tests"""
    logger.info("Starting ScienceQA dataset tests...")
    
    try:
        # Load dataset
        dataset = test_dataset_loading()
        
        # Run tests
        test_dataset_statistics(dataset)
        test_collate_function(dataset)
        test_edge_cases(dataset)
        
        logger.info("🎉 All ScienceQA dataset tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 