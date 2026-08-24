import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from competitive_architectures.continual import (
    _class_indices,
    _class_order,
    _evaluate,
    _summarize_accuracy_matrix,
)


def test_class_order_is_seeded_and_exhaustive() -> None:
    first = _class_order(4)
    second = _class_order(4)
    assert first == second
    assert sorted(first) == list(range(10))


def test_class_indices_are_balanced_and_seeded() -> None:
    targets = [0] * 10 + [1] * 10 + [2] * 10
    first = _class_indices(targets, [0, 2], seed=7, maximum_per_class=4)
    second = _class_indices(targets, [0, 2], seed=7, maximum_per_class=4)
    assert first == second
    assert len(first) == 8
    assert sum(targets[index] == 0 for index in first) == 4
    assert sum(targets[index] == 2 for index in first) == 4
    assert len(set(first)) == len(first)


def test_continual_summary_metrics_use_full_accuracy_matrix() -> None:
    matrix = [
        [0.8, None, None],
        [0.5, 0.9, None],
        [0.4, 0.6, 0.7],
    ]
    average_incremental, final_average, forgetting, acquisition = (
        _summarize_accuracy_matrix(matrix)
    )
    assert round(average_incremental, 6) == round((0.8 + 0.7 + 0.5666667) / 3, 6)
    assert round(final_average, 6) == round((0.4 + 0.6 + 0.7) / 3, 6)
    assert round(forgetting, 6) == 0.35
    assert round(acquisition, 6) == 0.8


class _DiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Identity()

    def representations(self, features: torch.Tensor) -> torch.Tensor:
        return features


def test_task_aware_evaluation_masks_unrelated_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 99.0], [3.0, 1.0, 99.0]])
    labels = torch.tensor([1, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    model = _DiagnosticModel()
    full_accuracy, _ = _evaluate(model, loader, torch.device("cpu"))
    task_accuracy, _ = _evaluate(
        model,
        loader,
        torch.device("cpu"),
        allowed_classes=[0, 1],
    )
    assert full_accuracy == 0
    assert task_accuracy == 1
