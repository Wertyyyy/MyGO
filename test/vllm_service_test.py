import asyncio
import time
import logging
import yaml
import fire
import torch

from vllm_service.vllm_client import VLLMClient
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
                {"type": "text", "text": "Describe this image in detail."},
            ],
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
    ],
    "text_only": [
        {
            "role": "user",
            "content": "Hello, how are you? Please respond briefly.",
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
                {"type": "text", "text": "Compare these two images."},
            ],
        },
    ],
}


async def test_basic_inference(client: VLLMClient):
    logger.info("Testing basic inference...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
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


async def test_multiple_generation(client: VLLMClient):
    logger.info("Testing multiple generation...")

    conversation = TEST_CONVERSATIONS["image_text"]

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


async def test_weight_update(client: VLLMClient, config: dict):
    logger.info("Testing multiple weight updates...")

    model_path = config["vllm_service"]["model_impl"]["params"]["model_name_or_path"]

    try:
        from transformers import Qwen2VLForConditionalGeneration

        logger.info("Loading full model weights for multiple updates...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )

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

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vllm_config = config["vllm_service"]
    host = vllm_config.get("host", "localhost")
    server_port = vllm_config.get("server_port", 40000)
    nccl_port = vllm_config.get("nccl_port", 41000)

    logger.info(f"Connecting to VLLM service at {host}:{server_port}")

    client = VLLMClient(
        host=host,
        server_port=server_port,
        nccl_port=nccl_port,
    )

    try:
        await client.initialize()
        logger.info("Connected to VLLM service successfully!")

        await test_basic_inference(client)
        await test_multiple_generation(client)
        await test_nccl_initialization(client)
        await test_weight_update(client, config)

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
