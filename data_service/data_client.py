import logging
import time
from typing import List, Optional
import requests
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np

from data_service.typing.grpo_data import GRPOData
from utils.metrics import MetricsManager

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
            allowed_methods=["HEAD", "GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def initialize(self):
        self._check_server()
        logger.info("Data client initialized")

    def _check_server(self, total_timeout: float = 300.0, retry_interval: float = 2.0):
        """Check if the data server is available"""
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
        """
        Update the current step on the data server
        
        Args:
            step: Current training step
            
        Returns:
            True if successful, False otherwise
        """
        url = f"http://{self.host}:{self.port}/update_step/"
        
        request_data = {
            "step": step
        }

        try:
            response = self.session.post(
                url,
                json=request_data,
                timeout=600,
            )
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Step updated to {step}")
                return result.get("status") == "success"
            else:
                logger.error(f"Step update request failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error updating step: {e}")
            return False

    def fetch_data(self, step: int, metrics: Optional[MetricsManager] = None) -> List[List[GRPOData]]:
        """
        Fetch data from the data server for a specific step
        
        Args:
            step: Training step to fetch data for
            metrics: Optional metrics manager for data statistics
            
        Returns:
            List of lists of GRPOData for training
        """
        url = f"http://{self.host}:{self.port}/fetch_data/"
        
        request_data = {
            "step": step
        }

        response = self.session.post(
            url,
            json=request_data,
            timeout=1200,
        )
        
        if response.status_code == 200:
            response_data = response.json()
            raw_data = response_data["data"]
            
            # Convert raw JSON data to GRPOData objects
            converted_data = []
            for batch in raw_data:
                converted_batch = []
                for item in batch:
                    grpo_item = GRPOData.model_validate(item)
                    converted_batch.append(grpo_item)
                converted_data.append(converted_batch)

            if metrics:
                data_statistics(converted_data, metrics)
            
            return converted_data
        else:
            raise Exception(f"Data fetch request failed: {response.status_code}, {response.text}")

    def reset(self) -> bool:
        """
        Reset the data server state to initial conditions
        
        Returns:
            True if successful, False otherwise
        """
        url = f"http://{self.host}:{self.port}/reset/"

        try:
            response = self.session.post(url, timeout=600)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Server reset successful: {result.get('message')}")
                return result.get("status") == "success"
            else:
                logger.error(f"Server reset request failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error resetting server: {e}")
            return False

    def close(self):
        """Close the HTTP session"""
        if self.session:
            self.session.close()



def data_statistics(all_data: List[List[GRPOData]], metrics: MetricsManager):
    flatten_data: List[GRPOData] = []
    for grouped_data in all_data:
        flatten_data.extend(grouped_data)

    # Length statistics
    lengths = np.array([data.length for data in flatten_data])
    prompt_lengths = np.array([data.prompt_length for data in flatten_data])
    response_lengths = np.array([data.response_length for data in flatten_data])

    metrics.add("Data/length/total/mean", np.mean(lengths).item())
    metrics.add("Data/length/total/max", np.max(lengths).item())
    metrics.add("Data/length/total/min", np.min(lengths).item())
    metrics.add("Data/length/prompt/mean", np.mean(prompt_lengths).item())
    metrics.add("Data/length/prompt/max", np.max(prompt_lengths).item())
    metrics.add("Data/length/prompt/min", np.min(prompt_lengths).item())
    metrics.add("Data/length/response/mean", np.mean(response_lengths).item())
    metrics.add("Data/length/response/max", np.max(response_lengths).item())
    metrics.add("Data/length/response/min", np.min(response_lengths).item())

    # Finish reason statistics
    finish_reasons = [data.stop_reason for data in flatten_data]
    by_length = finish_reasons.count("length")

    metrics.add("Data/finish_reason/length", by_length / len(finish_reasons))

    # Reward statistics
    reward_names = list(flatten_data[0].rewards.keys())

    all_rewards = []
    for data in flatten_data:
        if data.rewards is not None:
            all_rewards.append(list(data.rewards.values()))

    if all_rewards:
        all_rewards = np.array(all_rewards)
        for rwd_idx, rwd_name in enumerate(reward_names):
            rwd_i = all_rewards[:, rwd_idx]
            metrics.add(f"Reward/{rwd_name}/mean", np.mean(rwd_i).item())
            metrics.add(f"Reward/{rwd_name}/std", np.std(rwd_i).item())

        if len(reward_names) > 1:
            total_rewards = np.sum(all_rewards, axis=1)
            metrics.add("Reward/total/mean", np.mean(total_rewards).item())
            metrics.add("Reward/total/std", np.std(total_rewards).item())
