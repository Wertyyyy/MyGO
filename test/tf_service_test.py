import asyncio
import time
import logging
import yaml
import fire

from tf_service.tf_client import TFClient
from data_service.typing.messages import encode_image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


async def test_inference(client: TFClient):
    logger.info("Testing inference...")

    for test_name, test_conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            results = await client.get_logprobs([test_conversation])

            if results and "batched_logprobs" in results and len(results["batched_logprobs"]) > 0:
                logprobs = results["batched_logprobs"][0]

                logger.info(f"  Logprobs count: {len(logprobs)}")
                logger.info(f"  Logprobs: {logprobs}")
            else:
                logger.error(f"  Testing {test_name} failed: no results returned")

        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


async def test_large_batch_inference(client: TFClient):
    """Test large batch inference"""
    logger.info("Testing large batch inference...")

    # Create large batch
    large_batch = list(TEST_CONVERSATIONS.values()) * 25

    try:
        logger.info(f"Sending {len(large_batch)} conversations for large batch inference")

        start_time = time.time()
        results = await client.get_logprobs(large_batch)
        end_time = time.time()

        if results and "batched_logprobs" in results:
            batched_logprobs = results["batched_logprobs"]

            logger.info(
                f"Large batch inference completed in {end_time - start_time:.2f} seconds"
            )
            logger.info(f"Returned {len(batched_logprobs)} results")
            logger.info(
                f"Average time per message: {(end_time - start_time) / len(large_batch):.3f} seconds"
            )
        else:
            logger.error("Large batch inference failed: no results returned")

    except Exception as e:
        logger.error(f"Large batch inference test failed: {str(e)}")


async def test_tf_service(config_path: str):
    """Test TF Service"""
    logger.info("Starting TF Service tests")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tf_config = config["tf_service"]
    host = tf_config.get("host", "localhost")
    port = tf_config.get("port", 42000)

    logger.info(f"Connecting to TF service at {host}:{port}")

    # Initialize client
    client = TFClient(host=host, port=port)

    try:
        await client.initialize()
        logger.info("Connected to TF service successfully!")

        # Run all tests
        await test_inference(client)
        await test_large_batch_inference(client)

        logger.info("TF Service tests completed successfully")

    except Exception as e:
        logger.error(f"TF Service tests failed: {str(e)}")
        raise
    finally:
        await client.close()
        logger.info("Client connection closed")


def main(config_path: str):
    asyncio.run(test_tf_service(config_path))


if __name__ == "__main__":
    fire.Fire(main)
