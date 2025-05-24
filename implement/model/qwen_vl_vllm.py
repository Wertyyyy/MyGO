import logging
import uuid
from typing import Dict

from data_service.typing.messages import Conversation, extract_images_from_conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VLLMModelImpl:
    def __init__(
        self,
        model_name_or_path: str,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool,
        max_model_len: int,
        enforce_eager: bool,
        mm_processor_kwargs: Dict[str, int],
        limit_mm_per_prompt: Dict[str, int],
    ):
        self.model_name_or_path = model_name_or_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.mm_processor_kwargs = mm_processor_kwargs
        self.limit_mm_per_prompt = limit_mm_per_prompt

        logger.info(f"Loading VLLM model: {model_name_or_path}")
        self._load_engine()

    def _load_engine(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=self.model_name_or_path,
                dtype="bfloat16",
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enable_prefix_caching=self.enable_prefix_caching,
                max_model_len=self.max_model_len,
                enforce_eager=self.enforce_eager,
                mm_processor_kwargs=self.mm_processor_kwargs,
                limit_mm_per_prompt=self.limit_mm_per_prompt,
            )
        )
        logger.info("VLLM engine loaded successfully")

    async def generate(
        self,
        conversation: Conversation,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request_id = str(uuid.uuid4().hex)

        tokenizer = await self.engine.get_tokenizer()
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prompt_data = {"prompt": prompt_text}
        prompt_images = extract_images_from_conversation(conversation)
        if prompt_images:
            prompt_data["multi_modal_data"] = {"image": prompt_images}

        results_generator = self.engine.generate(
            prompt=prompt_data,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return {
            "completions": [output.text for output in final_output.outputs],
            "finish_reasons": [output.finish_reason for output in final_output.outputs],
        }
