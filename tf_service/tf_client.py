from typing import List, Dict
import logging
import time
import asyncio
import aiohttp
from requests import ConnectionError

from data_service.typing.messages import Conversation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TFClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.session = None

    async def initialize(self):
        await self._check_server()
        await self._create_session()

    async def _create_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def _check_server(
        self, total_timeout: float = 180.0, retry_interval: float = 2.0
    ):
        url = f"http://{self.host}:{self.port}/health/"
        start_time = time.time()

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            logger.info("Server is up!")
                            return None
            except aiohttp.ClientError as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The Transformers server can't be reached at {self.host}:{self.port} after {total_timeout} "
                        "seconds. Make sure the server is running."
                    ) from exc

            logger.info(
                f"Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            await asyncio.sleep(retry_interval)

    async def get_logprobs(
        self,
        conversations: List[Conversation],
    ) -> Dict[str, List[List]]:
        url = f"http://{self.host}:{self.port}/infer/"

        async with self.session.post(
            url,
            json={"conversations": conversations},
            timeout=300,
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                text = await response.text()
                raise Exception(f"Request failed: {response.status}, {text}")

    async def close(self):
        if self.session:
            await self.session.close()
