"""Run the exploratory signed-pathway causal diagnosis."""

import json
from pathlib import Path

from competitive_architectures.pathway_diagnostics import run_pathway_diagnosis


def main() -> None:
    result = run_pathway_diagnosis(data_dir="data/cifar10", seed=31)
    destination = Path("results/pathway_diagnosis_seed31.json")
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
