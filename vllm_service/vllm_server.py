from typing import List, Dict
import logging
import time
import fire
import asyncio
import importlib.util
import os
import uuid
import traceback

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from data_service.typing.message import Conversation

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


class UpdateWeightsNCCLRequest(BaseModel):
    names: List[str]
    shapes: List[List[int]]
    dtypes: List[str]


class VLLMServer:
    def __init__(
        self,
        llm_params: Dict,
        host: str,
        nccl_port: int,
    ):
        self.host = host
        self.nccl_port = nccl_port
        self.pynccl = None

        from vllm import AsyncLLMEngine, AsyncEngineArgs

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**llm_params)
        )
        logger.info("VLLM engine loaded successfully")

    async def generate(self, request: GenerateRequest) -> dict:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=request.n,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
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

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return {
            "completions": [output.text for output in final_output.outputs],
            "finish_reasons": [output.finish_reason for output in final_output.outputs],
        }

    def _init_nccl_blocking(self):
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        start_time = time.time()
        pg = StatelessProcessGroup.create(
            host=self.host, port=self.nccl_port, rank=1, world_size=2
        )
        end_time = time.time()
        logger.debug(
            f"NCCL initialization completed in {end_time - start_time} seconds"
        )
        self.pynccl = PyNcclCommunicator(pg, device=torch.device("cuda:0"))

    async def init_nccl(self):
        logger.info("Initializing NCCL")
        await asyncio.to_thread(self._init_nccl_blocking)
        return {"message": "NCCL initialized successfully"}

    def _update_weights_nccl_blocking(
        self, names: List[str], shapes: List[List[int]], dtypes: List[str]
    ):
        logging.info("Thread: Updating weights via NCCL start")
        start_time = time.time()

        # Safe dtype mapping
        dtype_mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
        }

        for name, shape, dtype in zip(names, shapes, dtypes):
            logger.debug(
                f"Updating weight: {name} with shape: {shape} and dtype: {dtype}"
            )
            torch_dtype = dtype_mapping[dtype]
            weight = torch.empty(
                shape, dtype=torch_dtype, device=torch.device("cuda:0")
            )
            self.pynccl.broadcast(weight, src=0, stream=torch.cuda.current_stream())
            self.pynccl.group.barrier()
            self.engine.engine.model_executor.driver_worker.model_runner.model.load_weights(
                weights=[(name, weight)]
            )
            del weight

        end_time = time.time()
        logger.debug(f"Weights update completed in {end_time - start_time} seconds")

    async def update_weights_nccl(
        self, names: List[str], shapes: List[List[int]], dtypes: List[str]
    ):
        logger.info("Updating weights via NCCL")
        await asyncio.to_thread(
            self._update_weights_nccl_blocking, names, shapes, dtypes
        )
        return {"message": "Weights updated successfully"}


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


def main(config_path: str):
    # Load the Python config module
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    vllm_config = config_module.vllm_server

    # Handle GPU configuration
    gpu_id = vllm_config["gpu_id"]
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    elif gpu_id < 0 and gpu_id >= -torch.cuda.device_count():
        total_gpu_nums = torch.cuda.device_count()
        logging.info(f"Total GPUs: {total_gpu_nums}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + total_gpu_nums)
        logging.info(f"Using GPU {gpu_id + total_gpu_nums}")
    else:
        raise ValueError(f"Invalid GPU ID: {gpu_id}")

    # Create server with configuration from vllm_server
    server = VLLMServer(
        llm_params=vllm_config["llm_params"],
        host=vllm_config["host"],
        nccl_port=vllm_config["nccl_port"],
    )

    app = create_app(server)
    config = uvicorn.Config(
        app,
        host=vllm_config["host"],
        port=vllm_config["port"],
        timeout_keep_alive=7200,
    )
    server_instance = uvicorn.Server(config)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
