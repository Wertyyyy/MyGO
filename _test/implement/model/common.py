import importlib.util
import logging

import fire

from _test.test_data.convs import (
    TEST_CONVERSATIONS_PURE_TEXT,
    TEST_CONVERSATIONS_MULTIMODAL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(config_file: str):
    logger.info(f"Starting test with config: {config_file}")

    # Load module from file path directly
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract tf_server configuration
    tf_config = config_module.tf_server
    impl_path = tf_config["impl_path"]
    model_params = tf_config["model_params"]
    processor_params = tf_config["processor_params"]
    
    # Load the implementation module
    impl_module = importlib.import_module(impl_path)
    
    # Initialize model with parameters from config
    model_impl = impl_module.TFModelImpl(**model_params)
    logger.info(f"✓ Model loaded - Multimodal: {model_impl.multimodal}")
    
    # Initialize processor with parameters from config
    processor_impl = impl_module.TFProcessorImpl(**processor_params)
    logger.info(f"✓ Processor loaded - Multimodal: {processor_impl.multimodal}")
    logger.info(f"Prefix IDs: {processor_impl.prefix_ids}")
    logger.info(f"Prefix Sequence: {repr(processor_impl.processor.decode(processor_impl.prefix_ids))}")

    if processor_impl.multimodal:
        test_conversations = TEST_CONVERSATIONS_MULTIMODAL
        logger.info(
            f"\nUsing multimodal conversations ({len(test_conversations)} cases)"
        )
    else:
        test_conversations = TEST_CONVERSATIONS_PURE_TEXT
        logger.info(
            f"\nUsing text-only conversations ({len(test_conversations)} cases)"
        )

    # Test different batch sizes
    for name, conversation in test_conversations.items():
        logger.info(f"Testing {name}")
        inputs = processor_impl.prepare_inputs([conversation])
        input_ids = inputs["input_ids"][0]
        real_seq_length = input_ids.shape[0]
        seq_length = processor_impl.get_seq_length(conversation)
        if seq_length != real_seq_length:
            logger.warning(f"{name}: {seq_length} != {real_seq_length}")
        else:
            logger.info(f"{name}: {seq_length}")

        logprobs = model_impl.get_batched_logprobs(inputs, processor_impl.prefix_ids)[0]

        total_length = input_ids.shape[0]
        response_length = logprobs.shape[0]
        prompt_length = total_length - response_length
        logger.info(
            f"  Total length: {total_length}, Prompt length: {prompt_length}, Response length: {response_length}"
        )

        # Get response tokens (last response_length tokens)
        response_tokens = input_ids[-response_length:].cpu().tolist()
        response_text = processor_impl.processor.decode(
            response_tokens, skip_special_tokens=False
        )
        logger.info(f"  Decoded response text : {repr(response_text)}")
        logger.info(f"  Original response text: {repr(conversation.messages[-1].content)}")

        # Show all token logprobs with their corresponding tokens
        for token_index, (token_id, logprob) in enumerate(
            zip(response_tokens, logprobs)
        ):
            token_text = processor_impl.processor.decode(
                [token_id], skip_special_tokens=False
            )
            logprob_val = logprob.item()
            logger.info(
                f"    [{token_index}] Token {token_id} ({repr(token_text)}): {logprob_val:.4f}"
            )


if __name__ == "__main__":
    fire.Fire(test)
