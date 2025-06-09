import time
import json

from data_service.data_client import DataClient


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
                print(f"  → Fetching data for step {step}")
                start_time = time.time()
                rank_data = data_client.fetch_data(
                    step, rank=0, update_step=False
                )
                end_time = time.time()
                print(f"  ✓ Finished in {end_time - start_time:.2f} seconds")
                print(f"  ✓ Fetched {len(rank_data)} micro steps")
                print(f"  ✓ Fetched {len(rank_data[0].data)} batches")
                # with open(f"step_{step}_list.json", "w") as f:
                #     json.dump(
                #         [[item.model_dump() for item in batch] for batch in data.data],
                #         f,
                #         indent=4,
                #     )
                # with open(f"step_{step}_grouped.json", "w") as f:
                #     json.dump(data.model_dump(), f, indent=4)
            except Exception as e:
                print(f"  ✗ Error fetching data: {e}")
                raise e
        else:
            print(f"  ✗ Failed to update to step {step}")

    # Clean up
    data_client.close()
    print(f"\n🎉 Training simulation completed! ({num_steps} steps)")


if __name__ == "__main__":
    main()
