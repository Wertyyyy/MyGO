import logging

from implement.model.qwen_tf import (
    TFServerModelImpl,
    DataServerProcessorImpl,
    MainTrainingImpl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/aiarena/project/2014-p15/models/Qwen2.5-3B-Instruct"  # You can change this to your local model path
MAX_LENGTH = 2048

TEST_CONVERSATIONS = {
    "case_1": [
        {
            "role": "user",
            "content": "Hello, how are you?",
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking!",
        },
    ],
    "case_2": [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
        {
            "role": "assistant",
            "content": "The capital of France is Paris.",
        },
    ],
    "case_3": [
        {
            "role": "user",
            "content": "Can you explain what machine learning is?",
        },
        {
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        },
    ],
    "case_4": [
        {
            "role": "user",
            "content": "Tell me about the history of artificial intelligence and its major milestones.",
        },
        {
            "role": "assistant",
            "content": "Artificial Intelligence has a rich history dating back to the 1950s. Key milestones include: 1950 - Alan Turing proposed the Turing Test; 1956 - The term 'Artificial Intelligence' was coined at the Dartmouth Conference; 1960s-70s - Early AI programs like ELIZA and expert systems; 1980s - Machine learning algorithms gained prominence; 1990s - Deep Blue defeated world chess champion Garry Kasparov; 2000s - Statistical machine learning became dominant; 2010s - Deep learning revolution with neural networks; 2020s - Large language models like GPT and modern AI assistants emerged.",
        },
    ],
    "case_5": [
        {
            "role": "user",
            "content": "Solve this math problem: What is 15 * 23 + 7 * 11?",
        },
        {
            "role": "assistant",
            "content": "Let me calculate this step by step:\n15 * 23 = 345\n7 * 11 = 77\n345 + 77 = 422\n\nTherefore, 15 * 23 + 7 * 11 = 422.",
        },
    ],
}


def test_sequence_length_tf_server(model: TFServerModelImpl):
    """Test sequence length calculation for TF Server model"""
    logger.info("Testing TF Server model sequence length calculation...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            length = model.get_seq_length(conversation)
            logger.info(f"  Sequence length: {length}")
        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


def test_sequence_length_data_processor(processor: DataServerProcessorImpl):
    """Test sequence length calculation for Data Server processor"""
    logger.info("Testing Data Server processor sequence length calculation...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            length = processor.get_seq_length(conversation)
            logger.info(f"  Sequence length: {length}")
        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


def test_single_processing(model: TFServerModelImpl):
    """Test single conversation processing"""
    logger.info("Testing single processing...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            results = model.process_batch([conversation], MAX_LENGTH)

            logprobs = results[0]

            logger.info(f"Single processing completed for {test_name}:")
            logger.info(f"  Logprobs count: {len(logprobs)}")
            if len(logprobs) > 0:
                logger.info(f"  First few logprobs: {logprobs[:5]}")
                logger.info(f"  Last few logprobs: {logprobs[-5:]}")
            else:
                logger.info("  No logprobs returned")

        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


def test_batch_processing(model: TFServerModelImpl):
    """Test batch processing of multiple conversations"""
    logger.info("Testing batch processing...")

    conversations = list(TEST_CONVERSATIONS.values())

    try:
        logger.info(f"Processing {len(conversations)} conversation sequences")
        results = model.process_batch(conversations, MAX_LENGTH)

        logger.info(f"Batch processing completed, returned {len(results)} results")

        for i, result in enumerate(results):
            test_name = list(TEST_CONVERSATIONS.keys())[i]
            logprobs = result

            logger.info(f"Result {i + 1} ({test_name}):")
            logger.info(f"  Logprobs count: {len(logprobs)}")
            if len(logprobs) > 0:
                logger.info(f"  First few logprobs: {logprobs[:3]}")
            else:
                logger.info("  No logprobs returned")

    except Exception as e:
        logger.error(f"Batch processing test failed: {str(e)}")


def test_main_training_impl(training_model: MainTrainingImpl):
    """Test MainTrainingImpl get_logprobs method"""
    logger.info("Testing MainTrainingImpl...")

    conversations = [TEST_CONVERSATIONS["case_1"], TEST_CONVERSATIONS["case_2"]]

    try:
        logger.info(f"Testing get_logprobs with {len(conversations)} conversations")
        results = training_model.get_logprobs(conversations, MAX_LENGTH)

        logger.info(f"MainTrainingImpl test completed, returned {len(results)} results")

        for i, result in enumerate(results):
            test_name = list(TEST_CONVERSATIONS.keys())[i]
            logger.info(f"Result {i + 1} ({test_name}):")
            logger.info(f"  Result type: {type(result)}")
            logger.info(
                f"  Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}"
            )
            logger.info(
                f"  Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}"
            )

    except Exception as e:
        logger.error(f"MainTrainingImpl test failed: {str(e)}")


def main():
    logger.info("Initializing models...")

    # Initialize TF Server model
    try:
        tf_model = TFServerModelImpl(
            model_name_or_path=MODEL_PATH,
            max_length=MAX_LENGTH,
        )
        logger.info("TF Server model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize TF Server model: {str(e)}")
        return

    # Initialize Data Server processor
    try:
        data_processor = DataServerProcessorImpl(
            model_name_or_path=MODEL_PATH,
        )
        logger.info("Data Server processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Data Server processor: {str(e)}")
        data_processor = None

    # Initialize Training model
    try:
        training_model = MainTrainingImpl(
            model_name_or_path=MODEL_PATH,
            max_length=MAX_LENGTH,
        )
        logger.info("Main Training model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Main Training model: {str(e)}")
        training_model = None

    # Run tests
    test_sequence_length_tf_server(tf_model)

    if data_processor:
        test_sequence_length_data_processor(data_processor)

    test_single_processing(tf_model)
    test_batch_processing(tf_model)

    if training_model:
        test_main_training_impl(training_model)

    logger.info("All tests completed")


if __name__ == "__main__":
    main()
