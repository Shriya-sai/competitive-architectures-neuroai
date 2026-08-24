from competitive_architectures.multiseed import _paired_summary


def _run(mode: str, seed: int, offset: float) -> dict[str, object]:
    return {
        "mode": mode,
        "seed": seed,
        "average_incremental_accuracy": 0.5 + offset,
        "final_average_accuracy": 0.4 + offset,
        "average_forgetting": 0.3 - offset,
        "mean_new_experience_accuracy": 0.7 + offset,
        "task_aware_final_accuracy": 0.8 + offset,
        "task_aware_forgetting": 0.2 - offset,
    }


def test_paired_summary_preserves_seed_pairing_and_contrast_direction() -> None:
    runs = []
    for seed in (3, 5, 7):
        runs.extend(
            (
                _run("standard", seed, 0.0),
                _run("random_signed", seed, 0.01),
                _run("structured_signed", seed, 0.03),
            )
        )
    summary = _paired_summary(runs, bootstrap_samples=100)
    contrast = summary["structured_minus_random"]["average_incremental_accuracy"]
    assert summary["seeds"] == [3, 5, 7]
    assert round(contrast["mean_difference"], 6) == 0.02
    assert contrast["structured_wins"] == 3
