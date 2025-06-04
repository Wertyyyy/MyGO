import logging
import time
import asyncio
import aiohttp
from requests import ConnectionError
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VLLMClient:
    def __init__(
        self,
        host: str,
        server_port: int,
        nccl_port: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.host = host
        self.server_port = server_port
        self.nccl_port = nccl_port
        self.max_retries = max_retries
        self.session = None

        self.loop = asyncio.new_event_loop()

    async def initialize(self):
        await self._check_server()
        await self._create_session()

    async def _create_session(self):
        """Create aiohttp session with improved connection pool configuration"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=600,
                connect=30,
                sock_read=300,
            )

            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            logger.info(
                f"Created new session for VLLM client {self.host}:{self.server_port}"
            )

    async def _check_server(
        self, total_timeout: float = 180.0, retry_interval: float = 2.0
    ):
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            logger.info("VLLM Server is up!")
                            return None
            except aiohttp.ClientError as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} seconds. "
                    ) from exc

            logger.info(
                f"VLLM Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            await asyncio.sleep(retry_interval)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                aiohttp.ClientError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
                ConnectionResetError,
            )
        ),
        reraise=True,
    )
    async def generate(
        self,
        conversation: Conversation,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Generate completions with automatic retry mechanism"""
        url = f"http://{self.host}:{self.server_port}/generate/"

        await self._create_session()

        try:
            logger.debug(f"Making generate request to {url}")

            async with self.session.post(
                url,
                json={
                    "conversation": conversation.model_dump(),
                    "n": n,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(
                        "Successfully received generate response from VLLM server"
                    )
                    return {
                        "completions": data["completions"],
                        "finish_reasons": data["finish_reasons"],
                    }
                else:
                    text = await response.text()
                    error_msg = (
                        f"VLLM generate request failed: {response.status}, {text}"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)

        except aiohttp.ServerDisconnectedError as e:
            logger.warning(f"VLLM server disconnected, will retry: {e}")
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            raise
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            logger.warning(f"VLLM client connection error, will retry: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in VLLM generate: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                aiohttp.ClientError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
                ConnectionResetError,
            )
        ),
        reraise=True,
    )
    async def init_nccl(self):
        """Initialize NCCL with automatic retry mechanism"""
        if self.nccl_port is None:
            raise ValueError("NCCL port is not set")

        url = f"http://{self.host}:{self.server_port}/init_nccl/"
        start_time = time.time()

        await self._create_session()

        try:
            # Step 1: First send the request to server to trigger server-side initialization
            logger.debug("Step 1: Sending NCCL initialization request to server...")
            send_request_task = asyncio.create_task(self.session.post(url))
            logger.debug("NCCL initialization request sent to server")

            # Step 2: Start client NCCL initialization in a separate thread
            logger.debug(
                "Step 2: Starting client NCCL initialization in a separate thread"
            )

            # Define the blocking NCCL initialization function
            def init_nccl_blocking():
                logger.debug("Thread: Starting NCCL initialization")
                pg = StatelessProcessGroup.create(
                    host=self.host, port=self.nccl_port, rank=0, world_size=2
                )
                pynccl = PyNcclCommunicator(pg, device="cuda:0")
                logger.debug("Thread: NCCL initialization completed")
                return pynccl

            # Run the blocking operation in a thread pool
            self.pynccl = await asyncio.to_thread(init_nccl_blocking)
            logger.debug("Client NCCL initialization completed")

            # Step 3: Now wait for the server's HTTP response
            logger.debug("Step 3: Waiting for server HTTP response...")
            response = await send_request_task
            logger.debug(f"Received server response with status: {response.status}")
            time_end = time.time()
            logger.debug(
                f"Time taken to initialize NCCL: {time_end - start_time} seconds"
            )

            if response.status == 200:
                data = await response.json()
                logger.debug(
                    f"Server NCCL initialization successful: {data['message']}"
                )
            else:
                text = await response.text()
                error_msg = (
                    f"Server NCCL initialization failed: {response.status}, {text}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

        except aiohttp.ServerDisconnectedError as e:
            logger.warning(
                f"VLLM server disconnected during NCCL init, will retry: {e}"
            )
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            raise
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            logger.warning(
                f"VLLM client connection error during NCCL init, will retry: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Error during NCCL initialization: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                aiohttp.ClientError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
                ConnectionResetError,
            )
        ),
        reraise=True,
    )
    async def update_weights_nccl(self, state_dict):
        """Update weights via NCCL with automatic retry mechanism"""
        time_start = time.time()
        url = f"http://{self.host}:{self.server_port}/update_weights_nccl/"

        await self._create_session()

        try:
            # Step 1: First prepare the weight information
            logger.debug("Step 1: Preparing weight information...")
            names = []
            shapes = []
            dtypes = []
            for name, p in state_dict.items():
                names.append(name)
                shapes.append(p.shape)
                dtypes.append(str(p.dtype))

            # Step 2: First send the request to server to trigger server-side preparation
            logger.debug("Step 2: Sending HTTP request to notify server...")
            send_request_task = asyncio.create_task(
                self.session.post(
                    url,
                    json={
                        "names": names,
                        "shapes": shapes,
                        "dtypes": dtypes,
                    },
                )
            )
            logger.debug("Weight update request sent to server")

            # Step 3: Start client NCCL weight transfer in a separate thread
            logger.debug(
                "Step 3: Starting client NCCL weight transfer in a separate thread"
            )

            # Define the blocking NCCL weight transfer function
            def send_weights_blocking(state_dict):
                logger.debug("Thread: Starting NCCL weight transfer")
                start_time = time.time()
                for name, p in state_dict.items():
                    # p = torch.zeros(p.shape, dtype=p.dtype, device="cuda:0")
                    self.pynccl.broadcast(p, src=0, stream=torch.cuda.current_stream())
                    self.pynccl.group.barrier()
                end_time = time.time()
                logger.debug(
                    f"Thread: NCCL weight transfer completed in {end_time - start_time} seconds"
                )

            # Run the blocking operation in a thread pool
            time_mid = time.time()
            await asyncio.to_thread(send_weights_blocking, state_dict)

            # Step 4: Now wait for the server's HTTP response
            logger.debug("Step 4: Waiting for server HTTP response...")
            response = await send_request_task
            time_end = time.time()
            logger.debug(f"Time taken to send weights: {time_end - time_mid} seconds")
            logger.debug(
                f"Time taken to receive response: {time_end - time_start} seconds"
            )

            if response.status == 200:
                data = await response.json()
                logger.debug(f"Server weights update successful: {data['message']}")
            else:
                text = await response.text()
                error_msg = f"Server weights update failed: {response.status}, {text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except aiohttp.ServerDisconnectedError as e:
            logger.warning(
                f"VLLM server disconnected during weight update, will retry: {e}"
            )
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            raise
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            logger.warning(
                f"VLLM client connection error during weight update, will retry: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Error during weights update: {str(e)}")
            raise

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("VLLM client session closed")

    def initialize_sync(self):
        self.loop.run_until_complete(self.initialize())

    def init_nccl_sync(self):
        self.loop.run_until_complete(self.init_nccl())

    def update_weights_nccl_sync(self, state_dict):
        self.loop.run_until_complete(self.update_weights_nccl(state_dict))

    def close_sync(self):
        self.loop.run_until_complete(self.close())
        self.loop.close()
