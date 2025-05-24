import logging
import time
import asyncio
import uuid
from typing import List, Any
import contextlib
import yaml
import importlib
import socket

import fire
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferenceRequest(BaseModel):
    conversations: List[List[dict]]


class InferenceResponse(BaseModel):
    batched_logprobs: List[List[float]]


class TransformersServer:
    def __init__(
        self,
        model_impl: dict[str, Any],
        token_budget: int,
    ):
        model_module = importlib.import_module(model_impl["path"])
        self.model = model_module.TFServerModelImpl(**model_impl["params"])
        self.token_budget = token_budget

        self.pending_requests = []
        self.request_lock = asyncio.Lock()
        self.results = {}
        self.processing_task = None

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
            first_msgs = first_req["conversation"]
            request_id = first_req["request_id"]
            seq_len = first_req["seq_len"]
            max_seq_len = seq_len
            current_token_usage = seq_len

            batch.append(first_msgs)
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

    async def _processing_loop(self):
        logger.info("Processing loop started")
        while True:
            batch, batch_ids = await self._fetch_batch()
            results = await asyncio.to_thread(self.model.process_batch, batch)
            for request_id, result in zip(batch_ids, results):
                self.results[request_id] = result

    async def add_requests(self, conversations):
        request_ids = []
        async with self.request_lock:
            for conversation in conversations:
                request_id = str(uuid.uuid4())

                # Pre-calculate sequence length
                seq_len = self.model.get_seq_length(conversation)

                # Store as dictionary with messages, request_id, and seq_len
                self.pending_requests.append(
                    {"conversation": conversation, "request_id": request_id, "seq_len": seq_len}
                )

                logger.info(f"Added request {request_id} to queue (seq_len: {seq_len})")
                request_ids.append(request_id)

        logger.info(f"Queue size: {len(self.pending_requests)}")
        return request_ids

    async def get_logprobs(self, request_ids: List[str], timeout: float = 60.0):
        start_time = time.time()
        while any(request_id not in self.results for request_id in request_ids):
            if time.time() - start_time > timeout:
                return None
            await asyncio.sleep(0.1)

        results = []
        for request_id in request_ids:
            results.append(self.results.pop(request_id))
        return results


def create_app(server: TransformersServer):
    app = FastAPI()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await server.start_processing_loop()
        yield

    app.router.lifespan_context = lifespan

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/infer/", response_model=InferenceResponse)
    async def infer(request: InferenceRequest):
        start_time = time.time()
        request_ids = await server.add_requests(request.conversations)
        results = await server.get_logprobs(request_ids)
        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds")

        return {
            "batched_logprobs": results,
        }

    return app


def main(
    config_file: str,
):
    with open(config_file, "r") as f:
        server_config = yaml.safe_load(f)["tf_service"]

    if server_config["host"] in ["localhost", "0.0.0.0"]:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"Server running on local IP: {local_ip}")

    server = TransformersServer(
        model_impl=server_config["model_impl"],
        token_budget=server_config["token_budget"],
    )

    app = create_app(server)

    config = uvicorn.Config(
        app,
        host=server_config["host"],
        port=server_config["port"],
        timeout_keep_alive=7200,
    )
    server_instance = uvicorn.Server(config)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
