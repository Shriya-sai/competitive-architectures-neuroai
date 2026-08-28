from competitive_architectures.development import _summarize


def test_development_summary_uses_paired_topology_contrasts() -> None:
    runs = []
    for index, seed in enumerate((23, 31, 47, 59, 71)):
        for mode, offset in (("random_signed", 0.0), ("structured_signed", 0.1)):
            run = {
                "seed": seed,
                "mode": mode,
                "post_signed_edge_tuning_gap": offset + index,
                "post_group_tuning_gap": offset + index,
            }
            for outcome in (
                "average_incremental_accuracy",
                "final_average_accuracy",
                "mean_new_experience_accuracy",
                "task_aware_final_accuracy",
            ):
                run[outcome] = offset + index
            for outcome in ("average_forgetting", "task_aware_forgetting"):
                run[outcome] = -offset + index
            runs.append(run)
    summary = _summarize(runs)
    accuracy = summary["structured_minus_random"]["average_incremental_accuracy"]
    forgetting = summary["structured_minus_random"]["average_forgetting"]
    assert abs(accuracy["mean_difference"] - 0.1) < 1e-12
    assert accuracy["structured_wins"] == 5
    assert abs(forgetting["mean_difference"] + 0.1) < 1e-12
