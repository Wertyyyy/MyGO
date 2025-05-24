import os

os.environ["VLLM_USE_V1"] = "0"

import asyncio
import logging
import time

from implement.model.qwen_vl_vllm import VLLMModelImpl
from data_service.typing.messages import encode_image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/aiarena/project/2014-p15/models/Qwen2-VL-Instruct"
GPU_MEMORY_UTILIZATION = 0.85
ENABLE_PREFIX_CACHING = False
MAX_MODEL_LEN = 8192
ENFORCE_EAGER = True
MM_PROCESSOR_KWARGS = {
    "min_pixels": 3136,
    "max_pixels": 401408,
}
LIMIT_MM_PER_PROMPT = {
    "image": 5,
    "video": 0,
}

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


async def test_basic_inference(model: VLLMModelImpl):
    logger.info("Testing basic inference...")

    for test_name, conversation in TEST_CONVERSATIONS.items():
        logger.info(f"Testing {test_name}")

        try:
            start_time = time.time()
            response = await model.generate(
                conversation=conversation,
                n=1,
                temperature=0.7,
                max_tokens=64,
            )
            end_time = time.time()

            logger.info(f"  Inference completed in {end_time - start_time:.2f} seconds")
            logger.info(f"  Generated text: {response['completions'][0][:80]}...")
            logger.info(f"  Finish reason: {response['finish_reasons'][0]}")
        except Exception as e:
            logger.error(f"  Testing {test_name} failed: {str(e)}")


async def test_multiple_generation(model: VLLMModelImpl):
    logger.info("Testing multiple generation...")

    conversation = TEST_CONVERSATIONS["image_text"]

    try:
        start_time = time.time()
        response = await model.generate(
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
            logger.info(f"  Response {i + 1}: {completion[:50]}...")
            logger.info(f"  Finish reason: {response['finish_reasons'][i]}")

    except Exception as e:
        logger.error(f"Multiple generation test failed: {str(e)}")


async def main():
    logger.info("Initializing VLLM model...")
    model = VLLMModelImpl(
        model_name_or_path=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=ENFORCE_EAGER,
        mm_processor_kwargs=MM_PROCESSOR_KWARGS,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    )
    logger.info("VLLM model initialized")

    await test_basic_inference(model)
    await test_multiple_generation(model)


if __name__ == "__main__":
    asyncio.run(main())
