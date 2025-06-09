import os

os.environ["VLLM_USE_V1"] = "1"

from typing import List
import logging
import time
import fire
import asyncio
import uuid
import traceback
from contextlib import ExitStack

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from data_service.typing.message import Conversation
from config.utils import ConfigManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerateRequest(BaseModel):
    conversation: Conversation
    n: int
    temperature: float
    max_tokens: int


class GenerateResponse(BaseModel):
    completions: List[str]
    finish_reasons: List[str]
    # For debugging
    prompt_token_ids: List[int]


class UpdateWeightsNCCLRequest(BaseModel):
    names: List[str]
    shapes: List[List[int]]
    dtypes: List[str]


class WorkerExtension:
    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        rank = get_world_group().rank + rank_offset
        logger.info(
            f"Worker initialized with rank {rank}, world_size {world_size}, device {self.device}"
        )

        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
        self.pynccl = PyNcclCommunicator(pg, device=self.device)

    def update_weight(
        self, names: List[str], shapes: List[List[int]], dtypes: List[str]
    ):
        import torch

        logging.info("Worker: Updating weights via NCCL start")
        start_time = time.time()

        dtype_mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
        }

        for name, shape, dtype in zip(names, shapes, dtypes):
            logger.debug(
                f"Updating weight: {name} with shape: {shape} and dtype: {dtype}"
            )
            torch_dtype = dtype_mapping[dtype]
            weight = torch.empty(
                shape,
                dtype=torch_dtype,
                device=torch.device(self.device),
            )
            self.pynccl.broadcast(weight, src=0, stream=torch.cuda.current_stream())
            self.model_runner.model.load_weights(weights=[(name, weight)])
            del weight

        end_time = time.time()
        logger.debug(f"Weights update completed in {end_time - start_time} seconds")


class VLLMServer:
    def __init__(
        self,
        config: ConfigManager,
    ):
        self.config = config
        self.host = "0.0.0.0"
        self.nccl_port = config.network.nccl_port
        self.pynccl = None
        self.stats_task = None

        from vllm import AsyncLLMEngine, AsyncEngineArgs

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                **config.vllm_server.llm_params.to_dict(),
                worker_extension_cls="vllm_service.vllm_server.WorkerExtension",
            )
        )
        logger.info("VLLM engine loaded successfully")

    async def _periodic_stats_logging(self):
        while True:
            await asyncio.sleep(1)
            await self.engine.do_log_stats()

    async def generate(self, request: GenerateRequest) -> dict:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        if self.stats_task is None:
            self.stats_task = asyncio.create_task(self._periodic_stats_logging())

        sampling_params = SamplingParams(
            n=request.n,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        request_id = str(uuid.uuid4().hex)

        tokenizer = await self.engine.get_tokenizer()
        prompt_text = tokenizer.apply_chat_template(
            request.conversation, tokenize=False, add_generation_prompt=True
        )
        prompt_data = {"prompt": prompt_text}
        prompt_images = request.conversation.get_images()
        if prompt_images:
            prompt_data["multi_modal_data"] = {"image": prompt_images}

        results_generator = self.engine.generate(
            prompt=prompt_data,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        completions = [[] for _ in range(request.n)]
        finish_reasons = [None for _ in range(request.n)]
        prompt_token_ids = None
        async for request_output in results_generator:
            if prompt_token_ids is None:
                prompt_token_ids = request_output.prompt_token_ids
            for output in request_output.outputs:
                completions[output.index] = output.text
                finish_reasons[output.index] = output.finish_reason

        return {
            "completions": completions,
            "finish_reasons": finish_reasons,
            "prompt_token_ids": prompt_token_ids,
        }

    async def init_nccl(self):
        logger.info("Received NCCL initialization request")

        # Start NCCL initialization in background without waiting
        asyncio.create_task(self._init_nccl_background())

        # Immediately return response to client
        return {"message": "NCCL initialization started"}

    async def _init_nccl_background(self):
        """Background task for NCCL initialization"""
        try:
            logger.info("Starting background NCCL initialization")
            await self.engine.collective_rpc(
                "init_weight_update_group",
                args=(
                    self.host,
                    self.nccl_port,
                    1,
                    self.config.vllm_server.llm_params.data_parallel_size + 1,
                ),
            )
            logger.info("Background NCCL initialization completed successfully")
        except Exception as e:
            logger.error(f"Background NCCL initialization failed: {e}")
            raise

    async def update_weights_nccl(
        self, names: List[str], shapes: List[List[int]], dtypes: List[str]
    ):
        logger.info("Updating weights via NCCL")
        await self.engine.collective_rpc(
            "update_weight",
            args=(names, shapes, dtypes),
        )
        # await self.engine.reset_prefix_cache()
        return {"message": "Weights updated successfully"}

    def cleanup(self):
        """Clean up resources"""
        if self.stats_task and not self.stats_task.done():
            self.stats_task.cancel()
        self.engine.shutdown()


def create_app(server: VLLMServer):
    app = FastAPI()

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        try:
            return await server.generate(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error during generation: {str(e)}\nTraceback:\n{tb_str}")
            return GenerateResponse(
                completions=[f"Error: {str(e)}"],
                finish_reasons=["error"],
            )

    @app.post("/init_nccl/")
    async def init_nccl():
        return await server.init_nccl()

    @app.post("/update_weights_nccl/")
    async def update_weights_nccl(request: UpdateWeightsNCCLRequest):
        return await server.update_weights_nccl(
            request.names, request.shapes, request.dtypes
        )

    return app


def main(config_file: str):
    config = ConfigManager(config_file)

    # Set CUDA_VISIBLE_DEVICES based on device configuration
    devices = config.vllm_server.devices
    if isinstance(devices, list):
        gpu_indices = []
        for device in devices:
            gpu_index = device.split(":")[1]
            gpu_indices.append(gpu_index)

        cuda_visible_devices = ",".join(gpu_indices)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES to: {cuda_visible_devices}")

    server = VLLMServer(config=config)

    app = create_app(server)
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=config.network.vllm_port,
        timeout_keep_alive=7200,
        access_log=False,
    )
    server_instance = uvicorn.Server(uvicorn_config)

    with ExitStack() as after_stack:
        after_stack.callback(server.cleanup)
        asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
