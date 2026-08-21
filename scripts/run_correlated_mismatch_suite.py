"""Run the final correlated-feature and topology-mismatch validation."""

import json

from competitive_architectures.synthetic import run_correlated_mismatch_suite


def main() -> None:
    summaries = [result.to_dict() for result in run_correlated_mismatch_suite()]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
