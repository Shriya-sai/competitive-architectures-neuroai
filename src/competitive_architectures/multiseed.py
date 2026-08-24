"""Frozen paired-seed execution and aggregation for continual learning."""

import json
from itertools import product
from pathlib import Path

import numpy as np

from competitive_architectures.continual import run_split_cifar10_pilot

METRICS = (
    "average_incremental_accuracy",
    "final_average_accuracy",
    "average_forgetting",
    "mean_new_experience_accuracy",
    "task_aware_final_accuracy",
    "task_aware_forgetting",
)


def _paired_summary(
    runs: list[dict[str, object]],
    bootstrap_seed: int = 20260824,
    bootstrap_samples: int = 10000,
) -> dict[str, object]:
    by_mode = {
        mode: sorted(
            (run for run in runs if run["mode"] == mode),
            key=lambda run: int(run["seed"]),
        )
        for mode in ("standard", "random_signed", "structured_signed")
    }
    seed_lists = [
        [int(run["seed"]) for run in mode_runs] for mode_runs in by_mode.values()
    ]
    if not seed_lists or any(seeds != seed_lists[0] for seeds in seed_lists[1:]):
        raise ValueError("every condition must contain the same paired seeds")

    mode_means = {
        mode: {
            metric: float(np.mean([float(run[metric]) for run in mode_runs]))
            for metric in METRICS
        }
        for mode, mode_runs in by_mode.items()
    }
    random_runs = by_mode["random_signed"]
    structured_runs = by_mode["structured_signed"]
    contrasts = {}
    generator = np.random.default_rng(bootstrap_seed)
    for metric in METRICS:
        differences = np.array(
            [
                float(structured[metric]) - float(random[metric])
                for structured, random in zip(structured_runs, random_runs, strict=True)
            ]
        )
        sample_indices = generator.integers(
            0,
            len(differences),
            size=(bootstrap_samples, len(differences)),
        )
        bootstrap_means = differences[sample_indices].mean(axis=1)
        observed = abs(float(differences.mean()))
        sign_flip_means = np.array(
            [
                np.mean(differences * np.array(signs))
                for signs in product((-1, 1), repeat=len(differences))
            ]
        )
        contrasts[metric] = {
            "paired_differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
            "two_sided_exact_sign_flip_p": float(
                np.mean(np.abs(sign_flip_means) >= observed)
            ),
            "structured_wins": int((differences > 0).sum()),
        }
    return {
        "seeds": seed_lists[0],
        "condition_means": mode_means,
        "structured_minus_random": contrasts,
    }


def run_frozen_multiseed(
    config_path: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Run every frozen seed, saving after each seed for interruption safety."""
    config = json.loads(Path(config_path).read_text())
    if config.get("status") != "frozen_before_confirmation":
        raise ValueError("configuration is not marked frozen")
    seeds = [int(seed) for seed in config["confirmation_seeds"]]
    if int(config["development_seed_excluded"]) in seeds:
        raise ValueError("development seed may not enter confirmation")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"config": config, "completed_seeds": [], "runs": []}
    for seed in seeds:
        print(f"Starting paired seed {seed}", flush=True)
        seed_results = run_split_cifar10_pilot(
            data_dir=data_dir,
            seed=seed,
            epochs_per_experience=int(config["epochs_per_experience"]),
            train_examples_per_class=int(config["train_examples_per_class"]),
            batch_size=int(config["batch_size"]),
            replay_examples_per_class=int(config["replay_examples_per_class"]),
        )
        payload["runs"].extend(result.to_dict() for result in seed_results)
        payload["completed_seeds"].append(seed)
        destination.write_text(json.dumps(payload, indent=2))
        print(f"Completed paired seed {seed}", flush=True)

    payload["summary"] = _paired_summary(payload["runs"])
    destination.write_text(json.dumps(payload, indent=2))
    return payload
