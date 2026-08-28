import torch

from competitive_architectures.drift import RepresentationSnapshot, compare_snapshots


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
