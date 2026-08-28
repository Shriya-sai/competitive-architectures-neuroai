"""Exploratory multi-seed screening for the engaged signed bottleneck."""

import json
from pathlib import Path

import numpy as np

from competitive_architectures.continual import run_split_cifar10_pilot
from competitive_architectures.pathway_diagnostics import run_pathway_diagnosis

DEVELOPMENT_SEEDS = (23, 31, 47, 59, 71)
OUTCOMES = (
    "average_incremental_accuracy",
    "final_average_accuracy",
    "average_forgetting",
    "mean_new_experience_accuracy",
    "task_aware_final_accuracy",
    "task_aware_forgetting",
)


def _spearman(first: list[float], second: list[float]) -> float:
    """Compute Spearman correlation without an additional dependency."""
    first_ranks = np.argsort(np.argsort(np.asarray(first)))
    second_ranks = np.argsort(np.argsort(np.asarray(second)))
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    by_seed_mode = {
        (int(run["seed"]), str(run["mode"])): run for run in runs
    }
    contrasts = {}
    for outcome in OUTCOMES:
        differences = [
            float(by_seed_mode[(seed, "structured_signed")][outcome])
            - float(by_seed_mode[(seed, "random_signed")][outcome])
            for seed in DEVELOPMENT_SEEDS
        ]
        contrasts[outcome] = {
            "paired_differences": differences,
            "mean_difference": float(np.mean(differences)),
            "structured_wins": int(sum(value > 0 for value in differences)),
        }

    structured = [
        by_seed_mode[(seed, "structured_signed")] for seed in DEVELOPMENT_SEEDS
    ]
    edge_gaps = [float(run["post_signed_edge_tuning_gap"]) for run in structured]
    group_gaps = [float(run["post_group_tuning_gap"]) for run in structured]
    incremental = [float(run["average_incremental_accuracy"]) for run in structured]
    return {
        "mean_post_signed_edge_tuning_gap": float(np.mean(edge_gaps)),
        "minimum_post_signed_edge_tuning_gap": float(np.min(edge_gaps)),
        "mean_post_group_tuning_gap": float(np.mean(group_gaps)),
        "minimum_post_group_tuning_gap": float(np.min(group_gaps)),
        "structured_minus_random": contrasts,
        "exploratory_spearman": {
            "edge_gap_vs_incremental_accuracy": _spearman(edge_gaps, incremental),
            "group_gap_vs_incremental_accuracy": _spearman(group_gaps, incremental),
        },
    }


def run_bottleneck_development_screen(
    data_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Run five exposed seeds, checkpointing after each paired seed."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": "exploratory_exposed_seed_bottleneck_screen",
        "seeds": list(DEVELOPMENT_SEEDS),
        "completed_seeds": [],
        "runs": [],
    }
    for seed in DEVELOPMENT_SEEDS:
        print(f"Starting bottleneck development seed {seed}", flush=True)
        continual = run_split_cifar10_pilot(
            data_dir=data_dir,
            seed=seed,
            epochs_per_experience=3,
            train_examples_per_class=2000,
            batch_size=128,
            replay_examples_per_class=200,
            pathway_mode="signed_bottleneck",
        )
        diagnosis = run_pathway_diagnosis(
            data_dir=data_dir,
            seed=seed,
            epochs_per_experience=3,
            train_examples_per_class=2000,
            batch_size=128,
            replay_examples_per_class=200,
            pathway_mode="signed_bottleneck",
        )
        diagnostic_by_mode = {
            str(item["mode"]): item for item in diagnosis["diagnostics"]
        }
        for result in continual:
            if result.mode == "standard":
                continue
            run = result.to_dict()
            alignment = diagnostic_by_mode[result.mode]["post_topology_alignment"]
            run["post_signed_edge_tuning_gap"] = alignment["signed_edge_tuning_gap"]
            run["post_group_tuning_gap"] = alignment["group_tuning_gap"]
            run["pathway_accuracy_effect"] = diagnostic_by_mode[result.mode][
                "accuracy_effect"
            ]
            run["pathway_prediction_disagreement"] = diagnostic_by_mode[result.mode][
                "prediction_disagreement"
            ]
            payload["runs"].append(run)
        payload["completed_seeds"].append(seed)
        destination.write_text(json.dumps(payload, indent=2))
        print(f"Completed bottleneck development seed {seed}", flush=True)
    payload["summary"] = _summarize(payload["runs"])
    destination.write_text(json.dumps(payload, indent=2))
    return payload
