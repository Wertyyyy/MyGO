import logging
import time
import asyncio
import uuid
from typing import List, Any
import contextlib
import importlib.util
import socket
import os
import traceback

import torch
import fire
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferenceRequest(BaseModel):
    conversations: List[Conversation]


class InferenceResponse(BaseModel):
    batched_logprobs: List[List[float]]
    batched_input_ids: List[List[int]]


class TransformersServer:
    def __init__(
        self,
        impl_path: str,
        model_params: dict[str, Any],
        processor_params: dict[str, Any],
        token_budget: int,
        max_length: int,
    ):
        # Load model and processor implementations
        impl_module = importlib.import_module(impl_path)

        # Initialize model
        self.model = impl_module.TFModelImpl(**model_params)

        # Initialize processor
        self.processor = impl_module.TFProcessorImpl(**processor_params)

        self.max_length = max_length
        self.token_budget = token_budget
        self.pending_requests = []
        self.request_lock = asyncio.Lock()
        self.results = {}
        self.processing_task = None

        logger.info(f"Loaded model from: {model_params['model_name_or_path']}")
        logger.info(f"Multimodal: {self.model.multimodal}")
        logger.info(f"Prefix IDs: {self.processor.prefix_ids}")

    async def start_processing_loop(self):
        logger.info("Starting processing loop")
        self.processing_task = asyncio.create_task(self._processing_loop())
        return self.processing_task

    async def _fetch_batch(self):
        batch = []
        batch_ids = []
        current_token_usage = 0
        max_seq_len = 0

        while len(self.pending_requests) == 0:
            await asyncio.sleep(0.1)

        # Lock while we're examining and modifying the pending requests
        async with self.request_lock:
            # Take first item to start the batch
            first_req = self.pending_requests.pop(0)
            first_conversation = first_req["conversation"]
            request_id = first_req["request_id"]
            seq_len = first_req["seq_len"]
            max_seq_len = seq_len
            current_token_usage = seq_len

            batch.append(first_conversation)
            batch_ids.append(request_id)

            # Keep track of which indices we'll remove
            to_remove = []

            # Scan through all remaining requests to find ones that fit
            for i, req in enumerate(self.pending_requests):
                # Get pre-calculated token length
                next_seq_len = req["seq_len"]

                # Calculate new max sequence length if we add this item
                new_max_seq_len = max(max_seq_len, next_seq_len)

                # Calculate new token usage with updated padding
                new_token_usage = new_max_seq_len * (len(batch) + 1)

                # Check if adding this item would exceed token budget
                if new_token_usage <= self.token_budget:
                    # Add the item to our batch
                    batch.append(req["conversation"])
                    batch_ids.append(req["request_id"])
                    max_seq_len = new_max_seq_len
                    current_token_usage = new_token_usage
                    to_remove.append(i)

            # Remove the items we took (in reverse order to maintain indices)
            for i in sorted(to_remove, reverse=True):
                self.pending_requests.pop(i)

        logger.info(
            f"Created batch with {len(batch)} items, token usage: {current_token_usage}/{self.token_budget}"
        )
        return batch, batch_ids

    def _process_batch(self, batch: List[Conversation]) -> List[List[float]]:
        """Process a batch of conversations and return logprobs"""
        try:
            with torch.inference_mode():
                # Prepare inputs using processor
                inputs = self.processor.prepare_inputs(batch, self.max_length)

                # Get logprobs from model
                outputs = self.model.forward(inputs)
                batched_resp_logits, batched_input_ids = (
                    self.processor.get_batched_resp_logits_and_input_ids(
                        inputs, outputs
                    )
                )
                logprobs_tensors = self.processor.get_batched_logprobs(
                    batched_resp_logits, batched_input_ids
                )

                # Convert tensors to lists of floats
                results = []
                for logprob_tensor, input_ids in zip(
                    logprobs_tensors, batched_input_ids
                ):
                    results.append(
                        {
                            "logprobs": logprob_tensor.cpu().tolist(),
                            "input_ids": input_ids.cpu().tolist(),
                        }
                    )

                logger.info(
                    f"Processed batch of {len(batch)} conversations, returned {len(results)} results"
                )
                return results

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error processing batch: {e}\nTraceback:\n{tb_str}")
            # Return empty results for all conversations in case of error
            return [[] for _ in batch]

    async def _processing_loop(self):
        logger.info("Processing loop started")
        while True:
            try:
                batch, batch_ids = await self._fetch_batch()
                results = await asyncio.to_thread(self._process_batch, batch)

                # Store results
                for request_id, result in zip(batch_ids, results):
                    self.results[request_id] = result

            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Error in processing loop: {e}\nTraceback:\n{tb_str}")
                # Continue processing even if one batch fails
                continue

    async def add_requests(self, conversations: List[Conversation]):
        request_ids = []
        async with self.request_lock:
            for conversation in conversations:
                request_id = str(uuid.uuid4())

                # Pre-calculate sequence length
                seq_len = self.processor.get_seq_length(conversation)

                # Store as dictionary with conversation, request_id, and seq_len
                self.pending_requests.append(
                    {
                        "conversation": conversation,
                        "request_id": request_id,
                        "seq_len": seq_len,
                    }
                )

                logger.info(f"Added request {request_id} to queue (seq_len: {seq_len})")
                request_ids.append(request_id)

        logger.info(f"Queue size: {len(self.pending_requests)}")
        return request_ids

    async def get_logprobs_and_input_ids(
        self, request_ids: List[str], timeout: float = 60.0
    ):
        start_time = time.time()
        while any(request_id not in self.results for request_id in request_ids):
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for results: {request_ids}")
                return None
            await asyncio.sleep(0.1)

        results = []
        for request_id in request_ids:
            results.append(self.results.pop(request_id))
        return results


def create_app(server: TransformersServer):
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start the processing loop
        await server.start_processing_loop()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/infer/", response_model=InferenceResponse)
    async def infer(request: InferenceRequest):
        # Add requests to the queue
        request_ids = await server.add_requests(request.conversations)

        # Wait for results
        results = await server.get_logprobs_and_input_ids(request_ids)

        if results is None:
            raise Exception("Timeout waiting for inference results")

        return InferenceResponse(
            batched_logprobs=[result["logprobs"] for result in results],
            batched_input_ids=[result["input_ids"] for result in results],
        )

    return app


def main(config_file: str):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(f"Local IP address: {local_ip}")

    # Load the Python config module
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    tf_config = config_module.tf_server
    model_config = config_module.model
    processor_config = config_module.processor

    # Handle GPU configuration
    gpu_id = tf_config["gpu_id"]
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

    # Create server
    server = TransformersServer(
        impl_path=model_config["impl_path"],
        model_params=model_config["params"],
        processor_params=processor_config["params"],
        token_budget=tf_config["token_budget"],
        max_length=tf_config["max_length"],
    )

    app = create_app(server)
    config = uvicorn.Config(
        app,
        host=tf_config["host"],
        port=tf_config["port"],
        timeout_keep_alive=7200,
    )
    server_instance = uvicorn.Server(config)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
