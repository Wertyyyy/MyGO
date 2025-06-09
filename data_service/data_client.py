import logging
import time
from typing import List

import requests
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_service.typing.grpo_data import BatchedGRPOData

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def initialize(self):
        self._check_server()
        logger.info("Data client initialized")

    def _check_server(self, total_timeout: float = 300.0, retry_interval: float = 2.0):
        url = f"http://{self.host}:{self.port}/health/"
        start_time = time.time()

        while True:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    logger.info("Data server is up!")
                    return
            except (requests.RequestException, ConnectionError) as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The Data server can't be reached at {self.host}:{self.port} after {total_timeout} "
                        "seconds. Make sure the server is running."
                    ) from exc

            logger.info(
                f"Data server is not up yet. Retrying in {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

    def update_step(self, step: int) -> bool:
        url = f"http://{self.host}:{self.port}/update_step/"

        request_data = {"step": step}

        try:
            response = self.session.post(
                url,
                json=request_data,
                timeout=600,
                proxies={"http": None, "https": None},
            )
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Step updated to {step}")
                return result.get("status") == "success"
            else:
                logger.error(
                    f"Step update request failed: {response.status_code}, {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error updating step: {e}")
            return False

    def fetch_data(
        self,
        step: int,
        rank: int,
        update_step: bool = True,
    ) -> List[BatchedGRPOData]:
        url = f"http://{self.host}:{self.port}/fetch_data/"

        request_data = {"step": step, "rank": rank, "update_step": update_step}

        response = self.session.post(
            url,
            json=request_data,
            timeout=1200,
            proxies={"http": None, "https": None},
        )

        if response.status_code == 200:
            response_data = response.json()
            raw_data = response_data["data"]
            rank_data_list = []
            for rank_data in raw_data:
                rank_data_list.append(BatchedGRPOData.model_validate(rank_data))
            return rank_data_list
        else:
            raise Exception(
                f"Data fetch request failed: {response.status_code}, {response.text}"
            )

    def reset(self) -> bool:
        url = f"http://{self.host}:{self.port}/reset/"

        try:
            response = self.session.post(
                url, timeout=600, proxies={"http": None, "https": None}
            )
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Server reset successful: {result.get('message')}")
                return result.get("status") == "success"
            else:
                logger.error(
                    f"Server reset request failed: {response.status_code}, {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error resetting server: {e}")
            return False

    def close(self):
        if self.session:
            self.session.close()
