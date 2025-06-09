import importlib.util
import logging

import fire
from tqdm import tqdm

from config.utils import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(config_file: str):
    logger.info(f"Starting test with config: {config_file}")

    config = ConfigManager(config_file)

    # Test training dataset
    logger.info("Testing training dataset...")
    train_dataset_module = importlib.import_module(config.dataset.train_impl_path)
    train_dataset = train_dataset_module.TrainingDatasetImpl(
        config.dataset.train_dataset_path,
        config.dataset.system_prompt_path,
        config.dataset.template_path,
    )

    # Test basic properties
    logger.info(f"Training dataset size: {len(train_dataset.dataset)}")
    assert len(train_dataset.dataset) > 0, "Training dataset should not be empty"

    sample_data, solution = train_dataset.collate_fn([train_dataset.dataset[0]])
    sample_data[0].pprint()
    logger.info(f"Solution: {solution[0]}")

    # Test dataset indexing
    for item in tqdm(train_dataset.dataset, desc="Testing training samples"):
        train_dataset.collate_fn([item])

    logger.info("✓ Training dataset test passed")

    # Test test dataset if configured
    if config.dataset.test_impl_path is not None:
        logger.info("Testing test dataset...")
        test_dataset_module = importlib.import_module(config.dataset.test_impl_path)
        test_dataset = test_dataset_module.TestDatasetImpl(
            config.dataset.test_dataset_path,
            config.dataset.system_prompt_path,
            config.dataset.template_path,
        )

        # Test basic properties
        logger.info(f"Test dataset size: {len(test_dataset.dataset)}")
        assert len(test_dataset.dataset) > 0, "Test dataset should not be empty"

        sample_data, solution = test_dataset.collate_fn([test_dataset.dataset[0]])
        sample_data[0].pprint()
        logger.info(f"Solution: {solution[0]}")

        # Test dataset indexing
        for item in tqdm(test_dataset.dataset, desc="Testing test samples"):
            test_dataset.collate_fn([item])

        logger.info("✓ Test dataset test passed")
    else:
        logger.info("No test dataset configured, skipping test dataset validation")

    logger.info("✓ All dataset tests passed")


if __name__ == "__main__":
    fire.Fire(test)
