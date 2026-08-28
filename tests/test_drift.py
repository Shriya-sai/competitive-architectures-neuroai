import torch

from competitive_architectures.drift import (
    RepresentationSnapshot,
    _intervention_summary,
    _normalized_group_profiles,
    _selective_summary,
    compare_snapshots,
)


def _snapshot(offset: float = 0.0) -> RepresentationSnapshot:
    return RepresentationSnapshot(
        centroids={0: torch.tensor([1.0, offset]), 1: torch.tensor([0.0, 1.0])},
        dispersions={0: 1.0 + offset, 1: 1.0},
        margins={0: 2.0 - offset, 1: 2.0},
        accuracies={0: 0.9 - offset, 1: 0.9},
        group_profiles={0: torch.tensor([1.0, offset]), 1: torch.tensor([0.0, 1.0])},
        classifier_rows=torch.tensor([[1.0, offset], [0.0, 1.0]]),
    )


def test_snapshot_comparison_separates_drift_and_decision_loss() -> None:
    result = compare_snapshots(_snapshot(), _snapshot(0.2), [0], [1])
    assert result["old_centroid_cosine_drift"] > 0
    assert result["old_group_profile_cosine_drift"] > 0
    assert result["old_classifier_cosine_drift"] > 0
    assert abs(result["old_dispersion_change"] - 0.2) < 1e-6
    assert abs(result["old_margin_change"] + 0.2) < 1e-6
    assert abs(result["old_accuracy_change"] + 0.2) < 1e-6


def test_unchanged_snapshot_has_zero_drift() -> None:
    snapshot = _snapshot()
    result = compare_snapshots(snapshot, snapshot, [0], [1])
    assert result["old_centroid_cosine_drift"] == 0
    assert result["old_group_profile_cosine_drift"] == 0
    assert result["old_classifier_cosine_drift"] == 0


def test_normalized_group_profiles_capture_relative_allocation() -> None:
    representations = torch.tensor([[1.0, 3.0, 2.0, 2.0]])
    groups = torch.tensor([0, 0, 1, 1])
    profiles = _normalized_group_profiles(representations, groups)
    assert torch.allclose(profiles, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(profiles.sum(dim=1), torch.ones(1))


def test_intervention_summary_uses_within_seed_control() -> None:
    runs = []
    for seed in (23, 31, 47, 59, 71):
        for mode in ("random_signed", "structured_signed"):
            for weight in (0.0, 10.0):
                value = weight / 10
                runs.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "profile_stability_weight": weight,
                        "mean_new_experience_accuracy": value,
                        "final_average_accuracy": value,
                        "transitions": [
                            {
                                "old_group_profile_cosine_drift": value,
                                "old_centroid_cosine_drift": value,
                                "old_margin_change": value,
                                "old_accuracy_change": value,
                            }
                        ],
                    }
                )
    summary = _intervention_summary(runs)
    effect = summary["regularized_minus_control"]["structured_signed"]
    assert effect["mean_group_drift"]["mean_difference"] == 1
    assert effect["final_average_accuracy"]["mean_difference"] == 1


def test_selective_summary_keeps_three_conditions_separate() -> None:
    runs = []
    for seed in (23, 31, 47, 59, 71):
        for mode in ("random_signed", "structured_signed"):
            for weight, selective, value in (
                (0.0, False, 0.0),
                (10.0, False, 1.0),
                (10.0, True, 2.0),
            ):
                runs.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "profile_stability_weight": weight,
                        "selective_consolidation": selective,
                        "mean_new_experience_accuracy": value,
                        "final_average_accuracy": value,
                        "transitions": [
                            {
                                "old_group_profile_cosine_drift": value,
                                "old_accuracy_change": value,
                            }
                        ],
                    }
                )
    summary = _selective_summary(runs)["structured_signed"]
    assert summary["control"]["final_average_accuracy"] == 0
    assert summary["global"]["final_average_accuracy"] == 1
    assert summary["selective"]["final_average_accuracy"] == 2
