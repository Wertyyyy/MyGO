import importlib.util
import logging

import fire

from _test.test_data.convs import (
    TEST_CONVERSATIONS_PURE_TEXT,
    TEST_CONVERSATIONS_MULTIMODAL,
)
from config.utils import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(config_file: str):
    logger.info(f"Starting test with config: {config_file}")

    config = ConfigManager(config_file)

    model_module = importlib.import_module(config.model.impl_path)
    init_params = config.model.init_params.to_dict()
    init_params["device_map"] = "cuda"
    model = model_module.TFModelImpl(init_params=init_params)
    logger.info("✓ Model loaded")

    processor_module = importlib.import_module(config.processor.impl_path)
    processor = processor_module.TFProcessorImpl(
        init_params=config.processor.init_params.to_dict()
    )
    logger.info(f"✓ Processor loaded - Multimodal: {processor.multimodal}")
    logger.info(f"Prefix IDs: {processor.prefix_ids}")
    logger.info(
        f"Prefix Sequence: {repr(processor.processor.decode(processor.prefix_ids))}"
    )

    if processor.multimodal:
        test_conversations = TEST_CONVERSATIONS_MULTIMODAL
        logger.info(
            f"\nUsing multimodal conversations ({len(test_conversations)} cases)"
        )
    else:
        test_conversations = TEST_CONVERSATIONS_PURE_TEXT
        logger.info(
            f"\nUsing text-only conversations ({len(test_conversations)} cases)"
        )

    # Add <think></think> prefix to all assistant responses for Qwen3
    logger.info("Adding prefix to assistant responses for Qwen3 testing")
    for name, conversation in test_conversations.items():
        for message in conversation.messages:
            if message.role == "assistant":
                message.content = "<think>\n\n</think>\n\n" + message.content

    # Test different batch sizes
    for name, conversation in test_conversations.items():
        logger.info(f"Testing {name}")

        inputs = processor.prepare_inputs([conversation])
        outputs = model.forward(inputs.to("cuda"))
        input_ids = inputs["input_ids"][0]
        real_seq_length = input_ids.shape[0]

        calc_seq_length = processor.get_seq_length(conversation)
        calc_prompt_token_ids, calc_response_token_ids = (
            processor.get_prompt_response_token_ids(conversation)
        )
        calc_prompt_length = len(calc_prompt_token_ids)
        calc_response_length = len(calc_response_token_ids)

        # Check if the calculated seq length from two methods are identical
        if calc_seq_length != calc_prompt_length + calc_response_length:
            logger.warning(
                "calc_seq_length != calc_prompt_length + calc_response_length"
            )
            logger.warning(
                f"{name}: {calc_seq_length} != {calc_prompt_length} + {calc_response_length}"
            )

        # Check if the calculated prompt and response length add up to the total length
        if calc_prompt_length + calc_response_length != real_seq_length:
            logger.warning(
                "calc_prompt_length + calc_response_length != real_seq_length"
            )
            logger.warning(
                f"{name}: {calc_prompt_length} + {calc_response_length} != {real_seq_length}"
            )

        # Check if the calculated sequence length is correct
        if calc_seq_length != real_seq_length:
            logger.warning("calc_seq_length != real_seq_length")
            logger.warning(f"{name}: {calc_seq_length} != {real_seq_length}")

        # Check if the calculated prompt token ids are correct
        if calc_prompt_token_ids != input_ids[:calc_prompt_length].cpu().tolist():
            logger.warning("calc_prompt_token_ids != input_ids[:calc_prompt_length]")
            logger.warning(
                f"{name}: {calc_prompt_token_ids} != {input_ids[:calc_prompt_length]}"
            )

        # Check if the calculated response token ids are correct
        if calc_response_token_ids != input_ids[calc_prompt_length:].cpu().tolist():
            logger.warning("calc_response_token_ids != input_ids[calc_prompt_length:]")
            logger.warning(
                f"{name}: {calc_response_token_ids} != {input_ids[calc_prompt_length:]}"
            )

        logprobs = processor.get_batched_response_logprobs(inputs, outputs)[0]

        # Check if the number of logprobs is correct
        if len(logprobs) != calc_response_length:
            logger.warning("len(logprobs) != calc_response_length")
            logger.warning(f"{name}: {len(logprobs)} != {calc_response_length}")

        logger.info(
            f"  Total length: {calc_seq_length}, "
            f"Prompt length: {calc_prompt_length}, "
            f"Response length: {calc_response_length}"
        )

        # Get response tokens (last response_length tokens)
        response_text = processor.processor.decode(
            calc_response_token_ids, skip_special_tokens=False
        )
        logger.info(f"  Decoded response text : {repr(response_text)}")
        logger.info(
            f"  Original response text: {repr(conversation.messages[-1].content)}"
        )

        # Show all token logprobs with their corresponding tokens
        for token_index, (token_id, logprob) in enumerate(
            zip(calc_response_token_ids, logprobs)
        ):
            token_text = processor.processor.decode(
                [token_id], skip_special_tokens=False
            )
            logprob_val = logprob.item()
            logger.info(
                f"    [{token_index}] Token {token_id} ({repr(token_text)}): {logprob_val:.4f}"
            )


if __name__ == "__main__":
    fire.Fire(test)
