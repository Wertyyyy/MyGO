import time
import json

from data_service.data_client import DataClient, data_statistics
from data_service.grouping import adaptive_grouping
from utils.metrics import MetricsManager


def main():
    # Create and initialize data client
    data_client = DataClient(host="localhost", port=43000)
    data_client.initialize()
    data_client.reset()

    print("🚀 Data client connected")

    # Run 15 training steps
    num_steps = 15
    print(f"\n📚 Running {num_steps} training steps...")

    for step in range(num_steps):
        print(f"\n→ Step {step}")

        # Update server to current step
        if data_client.update_step(step):
            print(f"  ✓ Updated to step {step}")

            # Give server time to pregenerate data
            # time.sleep(15)

            # Fetch training data
            try:
                metrics = MetricsManager(auto_log=False)
                print(f"  → Fetching data for step {step}")
                start_time = time.time()
                data = data_client.fetch_data(step, metrics)
                total_samples = sum(len(batch) for batch in data)
                end_time = time.time()
                print(
                    f"  ✓ Fetched {len(data)} batches with {total_samples} samples in {end_time - start_time:.2f} seconds"
                )
                with open(f"step_{step}_list.json", "w") as f:
                    json.dump(
                        [[item.model_dump() for item in batch] for batch in data],
                        f,
                        indent=4,
                    )
                data_statistics(data, metrics)
                grouped_data = adaptive_grouping(data, 7, 7000, 2, metrics)
                with open(f"step_{step}_grouped.json", "w") as f:
                    json.dump(grouped_data.model_dump(), f, indent=4)
                # grouped_data.log()
                metrics.gather_and_log(step=step)
            except Exception as e:
                print(f"  ✗ Error fetching data: {e}")
        else:
            print(f"  ✗ Failed to update to step {step}")

    # Clean up
    data_client.close()
    print(f"\n🎉 Training simulation completed! ({num_steps} steps)")


if __name__ == "__main__":
    main()
