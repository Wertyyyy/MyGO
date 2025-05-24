import yaml
import fire
import time

from data_service.data_service import DataService
from utils import Metrics, print_gathered_metrics


def main(config_file: str):
    with open(config_file, "r") as f:
        service_config = yaml.safe_load(f)["data_service"]
    data_service = DataService(**service_config)
    for index in range(100):
        metrics = Metrics()
        future = data_service.fetch_data(metrics)
        time.sleep(10)
        global_step_data = future.result()  # noqa
        print_gathered_metrics([metrics], index)


if __name__ == "__main__":
    fire.Fire(main)
