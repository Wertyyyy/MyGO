import os

os.environ["VLLM_USE_V1"] = "0"

from typing import List, Any
import logging
import time
import fire
import asyncio
import yaml
import importlib
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerateRequest(BaseModel):
    conversation: List[dict]
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
        model_impl: dict[str, Any],
        host: str,
        nccl_port: int,
    ):
        self.host = host
        self.nccl_port = nccl_port
        self.pynccl = None

        logger.info(f"Importing model implementation: {model_impl}")
        model_module = importlib.import_module(model_impl["path"])
        self.model_impl = model_module.VLLMModelImpl(**model_impl["params"])

    async def generate(self, request: GenerateRequest) -> dict:
        return await self.model_impl.generate(
            conversation=request.conversation,
            n=request.n,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

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

        for name, shape, dtype in zip(names, shapes, dtypes):
            logger.debug(
                f"Updating weight: {name} with shape: {shape} and dtype: {dtype}"
            )
            weight = torch.empty(
                shape, dtype=eval(dtype), device=torch.device("cuda:0")
            )
            self.pynccl.broadcast(weight, src=0, stream=torch.cuda.current_stream())
            self.pynccl.group.barrier()
            self.model_impl.engine.engine.model_executor.driver_worker.model_runner.model.load_weights(
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
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            return {
                "completions": [f"Error: {str(e)}"],
                "finish_reasons": ["error"],
            }

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
    with open(config_path, "r") as f:
        server_config = yaml.safe_load(f)["vllm_service"]

    gpu_id = server_config["gpu_id"]
    if gpu_id is not None:
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
    else:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_count - 1)
                logging.info(f"Using GPU {device_count - 1}")

    server = VLLMServer(
        model_impl=server_config["model_impl"],
        host=server_config["host"],
        nccl_port=server_config["nccl_port"],
    )

    app = create_app(server)
    config = uvicorn.Config(
        app,
        host=server_config["host"],
        port=server_config["server_port"],
        timeout_keep_alive=7200,
    )
    server_instance = uvicorn.Server(config)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
