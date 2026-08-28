"""Run the frozen five-seed group-profile stability intervention."""

import json

from competitive_architectures.drift import run_profile_stability_replication


def main() -> None:
    result = run_profile_stability_replication(
        data_dir="data/cifar10",
        output_path="results/group_profile_stability_replication.json",
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
