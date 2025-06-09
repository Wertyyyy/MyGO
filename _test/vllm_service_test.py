import asyncio
import time
import logging
import importlib

import fire
import torch

from vllm_service.vllm_client import VLLMClient
from vllm_service.nccl_client import NCCLClient
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
                max_tokens=1024,
            )
            end_time = time.time()

            logger.info(f"  Inference completed in {end_time - start_time:.2f} seconds")
            logger.info(f"  Generated text: {repr(response['completions'][0])}")
            logger.info(f"  Finish reason: {response['finish_reasons'][0]}")
            logger.info(f"  Prompt token IDs: {response['prompt_token_ids']}")
        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


async def test_multiple_generation(client: VLLMClient, test_conversations):
    logger.info("Testing multiple generation...")

    # Choose the first available conversation for multiple generation test
    conversation_name = list(test_conversations.keys())[0]
    conversation = test_conversations[conversation_name]

    logger.info(
        f"Using conversation '{conversation_name}' for multiple generation test"
    )

    try:
        start_time = time.time()
        response = await client.generate(
            conversation=conversation,
            n=3,
            temperature=0.8,
            max_tokens=1024,
        )
        end_time = time.time()

        logger.info(
            f"Multiple generation completed in {end_time - start_time:.2f} seconds"
        )
        logger.info(f"Generated {len(response['completions'])} responses")

        for i, completion in enumerate(response["completions"]):
            logger.info(f"  Response {i + 1}: {repr(completion)}")
            logger.info(f"  Finish reason: {response['finish_reasons'][i]}")
    except Exception as e:
        logger.error(f"Multiple generation test failed: {str(e)}")


async def test_real_data_inference_batch(
    client: VLLMClient,
    config: ConfigManager,
    n_samples: int = 32,
    max_samples: int = 50,
):
    """
    Test real data inference with batch processing - process all samples concurrently
    """
    logger.info("Testing real data inference with batch processing...")

    # Load dataset using the same approach as data_server.py
    dataset_module = importlib.import_module(config.dataset.train_impl_path)
    dataset = dataset_module.TrainingDatasetImpl(
        dataset_path=config.dataset.train_dataset_path,
        system_prompt_path=config.dataset.system_prompt_path,
        template_path=config.dataset.template_path,
    )

    logger.info("Dataset loaded successfully")
    logger.info(f"Dataset size: {len(dataset.dataset)}")
    logger.info(f"Will test with {min(max_samples, len(dataset.dataset))} samples")
    logger.info(f"Generating {n_samples} responses per sample")

    # Prepare all samples
    samples_to_process = []
    for i, item in enumerate(dataset.dataset):
        if i >= max_samples:
            break

        try:
            conversations, solutions = dataset.collate_fn([item])
            conversation = conversations[0]
            solution = solutions[0]
            samples_to_process.append((i, conversation, solution))
        except Exception as e:
            logger.error(f"Failed to prepare sample {i + 1}: {str(e)}")
            continue

    logger.info(f"Prepared {len(samples_to_process)} samples")

    # Process all samples concurrently with staggered delays
    async def process_single_sample(sample_data, delay_seconds):
        i, conversation, solution = sample_data
        try:
            # Add delay before starting this sample
            await asyncio.sleep(delay_seconds)
            # logger.info(f"Processing sample {i + 1}...")
            start_time = time.time()
            response = await client.generate(
                conversation=conversation,
                n=n_samples,
                temperature=0.7,
                max_tokens=config.length.max_response_length
                if hasattr(config, "length")
                else 1024,
            )
            end_time = time.time()

            return {
                "success": True,
                "sample_id": i + 1,
                "solution": solution,
                "conversation": conversation,
                "response": response,
                "inference_time": end_time - start_time,
                "tokens_generated": sum(
                    len(comp.split()) for comp in response["completions"]
                ),
            }
        except Exception as e:
            logger.error(f"Sample {i + 1} failed: {str(e)}")
            return {"success": False, "sample_id": i + 1, "error": str(e)}

    # Execute all tasks concurrently with 0.1s delay between each start
    total_start_time = time.time()
    results = await asyncio.gather(
        *[process_single_sample(sample, idx * 0.1) for idx, sample in enumerate(samples_to_process)]
    )
    total_end_time = time.time()

    # Process results
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    total_tokens_generated = sum(
        r.get("tokens_generated", 0) for r in successful_results
    )
    total_time = total_end_time - total_start_time

    # Log detailed results for successful samples
    for result in successful_results[:3]:
        logger.info(f"\n--- Sample {result['sample_id']} Results ---")
        logger.info(f"Solution: {result['solution']}")
        logger.info(f"Inference time: {result['inference_time']:.2f} seconds")
        logger.info(f"Generated {len(result['response']['completions'])} responses")
        for j, completion in enumerate(result["response"]["completions"][:3]):
            logger.info(f"  Response {j + 1}: {repr(completion)}")
            logger.info(f"  Finish reason: {result['response']['finish_reasons'][j]}")

    # Log failed samples
    for result in failed_results[:3]:
        logger.error(f"Sample {result['sample_id']} failed: {result['error']}")

    if failed_results:
        logger.warning(f"Some inferences failed ({len(failed_results)}/{len(results)})")
    else:
        logger.info("✓ All batch real data inferences completed successfully")

    logger.info("\n=== Batch Real Data Inference Test Summary ===")
    logger.info(f"Total samples processed: {len(results)}")
    logger.info(f"Successful inferences: {len(successful_results)}")
    logger.info(f"Failed inferences: {len(failed_results)}")
    logger.info(f"Total time taken: {total_time:.2f} seconds")
    logger.info(f"Average time per sample: {total_time / len(results):.2f} seconds")
    logger.info(f"Approximate total tokens generated: {total_tokens_generated}")
    logger.info(f"Throughput: {total_tokens_generated / total_time:.2f} tokens/s")


def test_nccl_initialization(nccl_client: NCCLClient):
    logger.info("Testing NCCL initialization...")

    try:
        start_time = time.time()
        nccl_client.init_nccl()
        end_time = time.time()

        logger.info(
            f"NCCL initialization completed in {end_time - start_time:.2f} seconds"
        )
        logger.info("NCCL communication established, can perform weight updates")
    except Exception as e:
        logger.error(f"NCCL initialization failed: {str(e)}")
        raise


def test_weight_update(nccl_client: NCCLClient, config: ConfigManager):
    logger.info("Testing multiple weight updates...")

    try:
        impl_module = importlib.import_module(config.model.impl_path)
        init_params = config.model.init_params.to_dict()
        init_params["device_map"] = torch.device(config.tf_server.device)
        model = impl_module.TFModelImpl(init_params=init_params).model

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
                nccl_client.update_weights_nccl(state_dict)
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

    # Use ConfigManager instead of directly loading config module
    config = ConfigManager(config_path)

    # Check multimodal support
    is_multimodal = check_multimodal_support(config)
    logger.info(f"Model multimodal support: {is_multimodal}")

    # Select appropriate test conversations
    if is_multimodal:
        test_conversations = TEST_CONVERSATIONS_MULTIMODAL
        logger.info(
            f"Using multimodal test conversations ({len(test_conversations)} cases)"
        )
    else:
        test_conversations = TEST_CONVERSATIONS_PURE_TEXT
        logger.info(
            f"Using text-only test conversations ({len(test_conversations)} cases)"
        )

    client = VLLMClient(
        host=config.network.vllm_host,
        server_port=config.network.vllm_port,
    )
    
    nccl_client = NCCLClient(
        host=config.network.vllm_host,
        server_port=config.network.vllm_port,
        nccl_port=config.network.nccl_port,
        nccl_device="cuda:0",
        dp_size=config.vllm_server.llm_params.data_parallel_size,
    )

    try:
        await client.initialize()
        logger.info("Connected to VLLM service successfully!")

        await test_basic_inference(client, test_conversations)
        await test_multiple_generation(client, test_conversations)
        # await test_real_data_inference_batch(client, config)
        test_nccl_initialization(nccl_client)
        test_weight_update(nccl_client, config)

        logger.info("VLLM Service tests completed successfully")

    except Exception as e:
        logger.error(f"VLLM Service tests failed: {str(e)}")
        raise
    finally:
        await client.close()
        nccl_client.close()
        logger.info("Client connections closed")


def main(config_path: str):
    asyncio.run(test_vllm_service(config_path))


if __name__ == "__main__":
    fire.Fire(main)
