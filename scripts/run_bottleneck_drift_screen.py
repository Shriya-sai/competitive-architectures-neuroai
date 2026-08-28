"""Run the five-seed bottleneck representational-drift screen."""

import json

from competitive_architectures.drift import run_drift_screen


def main() -> None:
    result = run_drift_screen(
        data_dir="data/cifar10",
        output_path="results/bottleneck_drift_screen.json",
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
