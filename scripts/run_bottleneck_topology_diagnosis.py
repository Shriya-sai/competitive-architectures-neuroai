"""Apply validated topology-alignment measures to the signed bottleneck."""

import json
from pathlib import Path

from competitive_architectures.pathway_diagnostics import run_pathway_diagnosis


def main() -> None:
    result = run_pathway_diagnosis(
        data_dir="data/cifar10",
        seed=23,
        pathway_mode="signed_bottleneck",
    )
    result["status"] = "exploratory_bottleneck_topology_diagnosis"
    destination = Path("results/bottleneck_topology_diagnosis_seed23.json")
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
