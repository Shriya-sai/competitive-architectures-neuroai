"""Run the five-seed signed-bottleneck development screen."""

import json

from competitive_architectures.development import run_bottleneck_development_screen


def main() -> None:
    result = run_bottleneck_development_screen(
        data_dir="data/cifar10",
        output_path="results/bottleneck_development_screen.json",
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
