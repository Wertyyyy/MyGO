import asyncio
import time
import logging
import importlib.util
import importlib
import fire
import torch

from vllm_service.vllm_client import VLLMClient
from _test.test_data.convs import TEST_CONVERSATIONS_MULTIMODAL, TEST_CONVERSATIONS_PURE_TEXT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_multimodal_support(config_module):
    """Check if the model implementation supports multimodal"""
    try:
        tf_config = config_module.tf_server
        impl_module = importlib.import_module(tf_config["impl_path"])
        return impl_module.TFModelImpl.multimodal
    except Exception as e:
        logger.warning(f"Could not check multimodal support: {e}")
        return False


async def test_basic_inference(client: VLLMClient, test_conversations):
    logger.info("Testing basic inference...")

    for test_name, conversation in test_conversations.items():
        logger.info(f"Testing {test_name}")

        try:
            start_time = time.time()
            response = await client.generate(
                conversation=conversation,
                n=1,
                temperature=0.7,
                max_tokens=64,
            )
            end_time = time.time()

            logger.info(f"  Inference completed in {end_time - start_time:.2f} seconds")
            logger.info(f"  Generated text: {response['completions'][0]}")
            logger.info(f"  Finish reason: {response['finish_reasons'][0]}")
        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


async def test_multiple_generation(client: VLLMClient, test_conversations):
    logger.info("Testing multiple generation...")

    # Choose the first available conversation for multiple generation test
    conversation_name = list(test_conversations.keys())[0]
    conversation = test_conversations[conversation_name]
    
    logger.info(f"Using conversation '{conversation_name}' for multiple generation test")

    try:
        start_time = time.time()
        response = await client.generate(
            conversation=conversation,
            n=3,
            temperature=0.8,
            max_tokens=32,
        )
        end_time = time.time()

        logger.info(
            f"Multiple generation completed in {end_time - start_time:.2f} seconds"
        )
        logger.info(f"Generated {len(response['completions'])} responses")

        for i, completion in enumerate(response["completions"]):
            logger.info(f"  Response {i + 1}: {completion}")
            logger.info(f"  Finish reason: {response['finish_reasons'][i]}")
    except Exception as e:
        logger.error(f"Multiple generation test failed: {str(e)}")


async def test_nccl_initialization(client: VLLMClient):
    logger.info("Testing NCCL initialization...")

    try:
        start_time = time.time()
        await client.init_nccl()
        end_time = time.time()

        logger.info(
            f"NCCL initialization completed in {end_time - start_time:.2f} seconds"
        )
        logger.info("NCCL communication established, can perform weight updates")
    except Exception as e:
        logger.error(f"NCCL initialization failed: {str(e)}")
        raise


async def test_weight_update(client: VLLMClient, config_module):
    logger.info("Testing multiple weight updates...")

    try:
        tf_config = config_module.tf_server
        impl_module = importlib.import_module(tf_config["impl_path"])
        model = impl_module.TFModelImpl(**tf_config["model_params"]).model

        state_dict = model.state_dict()
        total_size_mb = sum(
            p.numel() * p.element_size() for p in state_dict.values()
        ) / (1024 * 1024)
        total_size_gb = total_size_mb / 1024

        logger.info(f"Transferring all {len(state_dict)} model parameters")
        logger.info(f"Transfer size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")

        del model
        torch.cuda.empty_cache()

        num_updates = 3
        total_transfer_time = 0

        for i in range(num_updates):
            logger.info(f"Updating weights {i + 1} times")

            try:
                start_time = time.time()
                await client.update_weights_nccl(state_dict)
                end_time = time.time()

                transfer_time = end_time - start_time
                total_transfer_time += transfer_time
                throughput = total_size_mb / transfer_time if transfer_time > 0 else 0

                logger.info(
                    f"  Update {i + 1} completed in {transfer_time:.2f} seconds"
                )
                logger.info(
                    f"  Transfer throughput: {throughput:.2f} MB/s ({throughput / 1024:.2f} GB/s)"
                )

            except Exception as e:
                logger.error(f"  Update {i + 1} failed: {str(e)}")
                raise

        avg_transfer_time = total_transfer_time / num_updates
        avg_throughput = (
            total_size_mb / avg_transfer_time if avg_transfer_time > 0 else 0
        )

        logger.info(f"Completed {num_updates} weight updates")
        logger.info(f"Total transfer time: {total_transfer_time:.2f} seconds")
        logger.info(f"Average transfer time: {avg_transfer_time:.2f} seconds")
        logger.info(
            f"Average transfer throughput: {avg_throughput:.2f} MB/s ({avg_throughput / 1024:.2f} GB/s)"
        )

        del state_dict
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Multiple weight updates test failed: {str(e)}")
        raise


async def test_vllm_service(config_path: str):
    logger.info("Starting VLLM Service tests")

    # Load the Python config module
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Check multimodal support
    is_multimodal = check_multimodal_support(config_module)
    logger.info(f"Model multimodal support: {is_multimodal}")
    
    # Select appropriate test conversations
    if is_multimodal:
        test_conversations = TEST_CONVERSATIONS_MULTIMODAL
        logger.info(f"Using multimodal test conversations ({len(test_conversations)} cases)")
    else:
        test_conversations = TEST_CONVERSATIONS_PURE_TEXT
        logger.info(f"Using text-only test conversations ({len(test_conversations)} cases)")

    vllm_config = config_module.vllm_server
    host = vllm_config["host"]
    server_port = vllm_config["port"]
    nccl_port = vllm_config["nccl_port"]

    logger.info(f"Connecting to VLLM service at {host}:{server_port}")

    client = VLLMClient(
        host=host,
        server_port=server_port,
        nccl_port=nccl_port,
    )

    try:
        await client.initialize()
        logger.info("Connected to VLLM service successfully!")

        await test_basic_inference(client, test_conversations)
        await test_multiple_generation(client, test_conversations)
        await test_nccl_initialization(client)
        await test_weight_update(client, config_module)

        logger.info("VLLM Service tests completed successfully")

    except Exception as e:
        logger.error(f"VLLM Service tests failed: {str(e)}")
        raise
    finally:
        await client.close()
        logger.info("Client connection closed")


def main(config_path: str):
    asyncio.run(test_vllm_service(config_path))


if __name__ == "__main__":
    fire.Fire(main)
