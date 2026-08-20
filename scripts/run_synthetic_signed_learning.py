"""Run the first synthetic signed-topology learning check."""

import json

from competitive_architectures.synthetic import run_synthetic_learning_experiment


def main() -> None:
    result = run_synthetic_learning_experiment()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
