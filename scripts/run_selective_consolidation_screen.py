"""Run the five-seed selective-consolidation comparison."""

import json

from competitive_architectures.drift import run_selective_consolidation_screen


def main() -> None:
    result = run_selective_consolidation_screen(
        data_dir="data/cifar10",
        output_path="results/selective_consolidation_screen.json",
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
