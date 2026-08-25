"""Compare signed-pathway engagement mechanisms on development seed 23."""

import json
from pathlib import Path

from competitive_architectures.pathway_diagnostics import run_pathway_diagnosis


def main() -> None:
    results = [
        run_pathway_diagnosis(
            data_dir="data/cifar10",
            seed=23,
            pathway_mode=pathway_mode,
        )
        for pathway_mode in (
            "weak_residual",
            "gated_residual",
            "signed_bottleneck",
        )
    ]
    payload = {
        "status": "exploratory_development_seed_pathway_engagement_suite",
        "seed": 23,
        "results": results,
    }
    destination = Path("results/pathway_engagement_seed23.json")
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
