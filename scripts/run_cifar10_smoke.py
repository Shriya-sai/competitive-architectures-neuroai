"""Run the first real-task trainability check."""

import json

from competitive_architectures.cifar import run_cifar10_smoke


def main() -> None:
    results = run_cifar10_smoke(data_dir="data/cifar10")
    print(json.dumps([result.to_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
