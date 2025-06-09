import logging
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


class NCCLClient:
    def __init__(
        self,
        host: str,
        server_port: int,
        nccl_port: int,
        nccl_device: str,
        dp_size: int,
        max_retries: int = 3,
    ):
        self.host = host
        self.server_port = server_port
        self.nccl_port = nccl_port
        self.nccl_device = nccl_device
        self.dp_size = dp_size
        self.max_retries = max_retries
        self.session = requests.Session()
        self.pynccl = None

        # Setup retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def init_nccl(self):
        url = f"http://{self.host}:{self.server_port}/init_nccl/"
        start_time = time.time()

        # Step 1: Send request to server to trigger server-side initialization
        logger.debug("Step 1: Sending NCCL initialization request to server...")
        response = self.session.post(
            url, timeout=600, proxies={"http": None, "https": None}
        )
        logger.debug(f"Received server response with status: {response.status_code}")

        # Initialize NCCL
        # Step 2: Start client NCCL initialization
        logger.debug("Step 2: Starting client NCCL initialization")
        logger.debug("Starting NCCL initialization")
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.nccl_port,
            rank=0,
            world_size=self.dp_size + 1,
        )
        self.pynccl = PyNcclCommunicator(pg, device=self.nccl_device)
        logger.debug("NCCL initialization completed")
        logger.debug("Client NCCL initialization completed")

        time_end = time.time()
        logger.debug(f"Time taken to initialize NCCL: {time_end - start_time} seconds")

        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Server NCCL initialization successful: {data['message']}")
        else:
            error_msg = f"Server NCCL initialization failed: {response.status_code}, {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def update_weights_nccl(self, state_dict):
        if self.pynccl is None:
            raise RuntimeError("NCCL not initialized. Call init_nccl() first.")

        time_start = time.time()
        url = f"http://{self.host}:{self.server_port}/update_weights_nccl/"

        # Step 1: Prepare the weight information
        logger.debug("Step 1: Preparing weight information...")
        names = []
        shapes = []
        dtypes = []
        for name, p in state_dict.items():
            names.append(name)
            shapes.append(p.shape)
            dtypes.append(str(p.dtype))

        # Step 2: Start client NCCL weight transfer
        logger.debug("Step 2: Starting client NCCL weight transfer")

        def send_weights_blocking(state_dict):
            logger.debug("Starting NCCL weight transfer")
            start_time = time.time()
            for name, p in state_dict.items():
                self.pynccl.broadcast(
                    p.to(torch.device(self.nccl_device)),
                    src=0,
                    stream=torch.cuda.current_stream(),
                )
            end_time = time.time()
            logger.debug(
                f"NCCL weight transfer completed in {end_time - start_time} seconds"
            )

        # Send weights via NCCL
        time_mid = time.time()
        send_weights_blocking(state_dict)

        # Step 3: Send HTTP request to notify server
        logger.debug("Step 3: Sending HTTP request to notify server...")
        response = self.session.post(
            url,
            json={
                "names": names,
                "shapes": shapes,
                "dtypes": dtypes,
            },
            timeout=600,
            proxies={"http": None, "https": None},
        )
        time_end = time.time()
        logger.debug(f"Time taken to send weights: {time_end - time_mid} seconds")
        logger.debug(f"Time taken to receive response: {time_end - time_start} seconds")

        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Server weights update successful: {data['message']}")
        else:
            error_msg = (
                f"Server weights update failed: {response.status_code}, {response.text}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    def close(self):
        if self.session:
            self.session.close()
            logger.info("NCCL client session closed")
