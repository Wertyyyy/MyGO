import logging

from implement.model.qwen_vl_tf import TFServerModelImpl
from data_service.typing.messages import encode_image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/aiarena/project/2014-p15/models/Qwen2-VL-Instruct"
MIN_PIXELS = 3136
MAX_PIXELS = 401408
MAX_LENGTH = 2048

TEST_CONVERSATIONS = {
    "image_text": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data:image/jpg;base64,"
                    + encode_image_to_base64("test/images/demo_1.jpg"),
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": "The image depicts a woman and her dog sitting on a sandy beach at sunset.",
        },
    ],
    "image_only": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data:image/jpg;base64,"
                    + encode_image_to_base64("test/images/demo_2.jpg"),
                },
            ],
        },
        {
            "role": "assistant",
            "content": "There is an apple in the image.",
        },
    ],
    "text_only": [
        {
            "role": "user",
            "content": "Hello, how are you?",
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking!",
        },
    ],
    "multi_image": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data:image/jpg;base64,"
                    + encode_image_to_base64("test/images/demo_1.jpg"),
                },
                {
                    "type": "image",
                    "image": "data:image/jpg;base64,"
                    + encode_image_to_base64("test/images/demo_2.jpg"),
                },
                {
                    "type": "text",
                    "text": "Compare these two images and describe their differences.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The first image shows a woman and her dog on a beach at sunset, while the second image shows an apple. These are completely different subjects and settings.",
        },
    ],
}


def test_sequence_length(model: TFServerModelImpl):
    """Test sequence length calculation"""
    logger.info("Testing sequence length calculation...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            length = model.get_seq_length(conversation)
            logger.info(f"  Sequence length: {length}")
        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


def test_single_processing(model: TFServerModelImpl):
    logger.info("Testing single processing...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            results = model.process_batch([conversation], MAX_LENGTH)

            logprobs = results[0]

            logger.info(f"Single processing completed for {test_name}:")
            logger.info(f"  Logprobs count: {len(logprobs)}")
            logger.info(f"  Logprobs: {logprobs}")

        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


def test_batch_processing(model: TFServerModelImpl):
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
            logger.info(f"  Logprobs: {logprobs}")

    except Exception as e:
        logger.error(f"Batch processing test failed: {str(e)}")


def main():
    logger.info("Initializing TF Server model...")
    model = TFServerModelImpl(
        model_name_or_path=MODEL_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info("TF Server model initialized")

    test_sequence_length(model)
    test_single_processing(model)
    test_batch_processing(model)

    logger.info("All tests completed")


if __name__ == "__main__":
    main()
