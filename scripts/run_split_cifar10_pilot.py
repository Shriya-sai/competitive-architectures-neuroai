"""Run the first exploratory class-incremental DNN experiment."""

import json

from competitive_architectures.continual import run_split_cifar10_pilot


def main() -> None:
    results = run_split_cifar10_pilot(data_dir="data/cifar10")
    print(json.dumps([result.to_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
