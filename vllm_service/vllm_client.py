import logging
import time
import asyncio
import aiohttp
from requests import ConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VLLMClient:
    def __init__(
        self,
        host: str,
        server_port: int,
        max_retries: int = 3,
    ):
        self.host = host
        self.server_port = server_port
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
                    return data
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

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("VLLM client session closed")
