"""Stress-test signed-topology recovery under controlled confounds."""

import json

from competitive_architectures.synthetic import run_synthetic_stress_suite


def main() -> None:
    summaries = [result.to_dict() for result in run_synthetic_stress_suite()]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
