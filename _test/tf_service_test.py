import asyncio
import time
import logging
import importlib
import fire

from tf_service.tf_client import TFClient
from config.utils import ConfigManager
from _test.test_data.convs import (
    TEST_CONVERSATIONS_MULTIMODAL,
    TEST_CONVERSATIONS_PURE_TEXT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_multimodal_support(config: ConfigManager):
    """Check if the model implementation supports multimodal"""
    try:
        impl_module = importlib.import_module(config.model.impl_path)
        # Create a temporary processor to check multimodal support
        processor = impl_module.TFProcessorImpl(
            init_params=config.processor.init_params.to_dict()
        )
        return processor.multimodal
    except Exception as e:
        logger.warning(f"Could not check multimodal support: {e}")
        return False


async def test_inference(client: TFClient, test_conversations):
    logger.info("Testing inference...")

    for test_name, test_conversation in test_conversations.items():
        logger.info(f"Testing {test_name}")

        try:
            results = await client.get_logprobs_and_input_ids([test_conversation])

            if (
                results
                and "batched_logprobs" in results
                and len(results["batched_logprobs"]) > 0
            ):
                logprobs = results["batched_logprobs"][0]

                logger.info(f"  Logprobs count: {len(logprobs)}")
                logger.info(f"  Logprobs: {logprobs}")
            else:
                logger.error(f"  Testing {test_name} failed: no results returned")

        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


async def test_large_batch_inference(client: TFClient, test_conversations):
    """Test large batch inference"""
    logger.info("Testing large batch inference...")

    # Create large batch
    large_batch = list(test_conversations.values()) * 25

    try:
        logger.info(
            f"Sending {len(large_batch)} conversations for large batch inference"
        )

        start_time = time.time()
        results = await client.get_logprobs_and_input_ids(large_batch)
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

    # Use ConfigManager instead of directly loading config module
    config = ConfigManager(config_path)

    # Check multimodal support
    is_multimodal = check_multimodal_support(config)
    logger.info(f"Model multimodal support: {is_multimodal}")

    # Select appropriate test conversations
    if is_multimodal:
        test_conversations = TEST_CONVERSATIONS_MULTIMODAL
        logger.info("Using multimodal test conversations")
    else:
        test_conversations = TEST_CONVERSATIONS_PURE_TEXT
        logger.info("Using text-only test conversations")

    host = "0.0.0.0"
    port = config.network.tf_port

    logger.info(f"Connecting to TF service at {host}:{port}")

    # Initialize client
    client = TFClient(host=host, port=port)

    try:
        await client.initialize()
        logger.info("Connected to TF service successfully!")

        # Run all tests
        await test_inference(client, test_conversations)
        await test_large_batch_inference(client, test_conversations)

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
