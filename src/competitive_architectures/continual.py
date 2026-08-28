"""Sequential class-incremental pilot for the core CIFAR models."""

from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from competitive_architectures.models import (
    PathwayMode,
    paired_models,
    trainable_parameter_count,
)


@dataclass(frozen=True)
class ContinualResult:
    mode: str
    seed: int
    replay_examples_per_class: int
    class_order: list[int]
    experiences: list[list[int]]
    accuracy_matrix: list[list[float | None]]
    task_aware_accuracy_matrix: list[list[float | None]]
    average_incremental_accuracy: float
    final_average_accuracy: float
    average_forgetting: float
    mean_new_experience_accuracy: float
    task_aware_final_accuracy: float
    task_aware_forgetting: float
    representation_norms: list[float]
    parameter_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _class_order(seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(10, generator=generator).tolist()


def _class_indices(
    targets: list[int],
    classes: list[int],
    seed: int,
    maximum_per_class: int | None = None,
) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    indices = []
    target_tensor = torch.tensor(targets)
    for class_label in classes:
        candidates = torch.where(target_tensor == class_label)[0]
        order = torch.randperm(candidates.numel(), generator=generator)
        candidates = candidates[order]
        if maximum_per_class is not None:
            candidates = candidates[:maximum_per_class]
        indices.extend(candidates.tolist())
    return indices


def _loader(
    dataset: Subset,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    allowed_classes: list[int] | None = None,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    norm_sum = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        representations = model.representations(images)
        logits = model.classifier(representations)
        if allowed_classes is not None:
            allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
            allowed[allowed_classes] = True
            logits = logits.masked_fill(~allowed, float("-inf"))
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
        norm_sum += float(representations.norm(dim=1).sum())
    return correct / total, norm_sum / total


def _summarize_accuracy_matrix(
    matrix: list[list[float | None]],
) -> tuple[float, float, float, float]:
    row_means = [
        float(np.mean([value for value in row if value is not None])) for row in matrix
    ]
    final_values = [value for value in matrix[-1] if value is not None]
    diagonal = [matrix[index][index] for index in range(len(matrix))]
    forgetting = []
    for experience in range(len(matrix) - 1):
        history = [
            matrix[time][experience]
            for time in range(experience, len(matrix))
            if matrix[time][experience] is not None
        ]
        forgetting.append(max(history) - history[-1])
    return (
        float(np.mean(row_means)),
        float(np.mean(final_values)),
        float(np.mean(forgetting)),
        float(np.mean(diagonal)),
    )


def run_split_cifar10_pilot(
    data_dir: str | Path,
    seed: int = 23,
    epochs_per_experience: int = 3,
    train_examples_per_class: int = 2000,
    batch_size: int = 128,
    replay_examples_per_class: int = 0,
    pathway_mode: PathwayMode = "weak_residual",
) -> list[ContinualResult]:
    """Run one paired five-experience Class-IL development sequence."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_data = datasets.CIFAR10(
        data_dir, train=True, download=False, transform=transform
    )
    test_data = datasets.CIFAR10(
        data_dir, train=False, download=False, transform=transform
    )
    class_order = _class_order(seed)
    experiences = [class_order[index : index + 2] for index in range(0, 10, 2)]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = []

    for mode, model in paired_models(seed, pathway_mode=pathway_mode).items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        accuracy_matrix: list[list[float | None]] = []
        task_aware_matrix: list[list[float | None]] = []
        representation_norms = []
        memory_indices: list[int] = []

        for experience_index, classes in enumerate(experiences):
            train_indices = _class_indices(
                train_data.targets,
                classes,
                seed=seed + 100 * experience_index,
                maximum_per_class=train_examples_per_class,
            )
            train_subset = Subset(train_data, train_indices)
            for epoch in range(epochs_per_experience):
                current_batch_size = (
                    batch_size // 2 if replay_examples_per_class else batch_size
                )
                train_loader = _loader(
                    train_subset,
                    current_batch_size,
                    seed=seed + 1000 * experience_index + epoch,
                    shuffle=True,
                )
                replay_batches = []
                if memory_indices:
                    replay_loader = _loader(
                        Subset(train_data, memory_indices),
                        batch_size - current_batch_size,
                        seed=seed + 10000 + 1000 * experience_index + epoch,
                        shuffle=True,
                    )
                    replay_batches = list(replay_loader)
                replay_iterator = cycle(replay_batches) if replay_batches else None
                model.train()
                for images, labels in train_loader:
                    if replay_iterator is not None:
                        replay_images, replay_labels = next(replay_iterator)
                        images = torch.cat((images, replay_images), dim=0)
                        labels = torch.cat((labels, replay_labels), dim=0)
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    loss = loss_fn(model(images), labels)
                    loss.backward()
                    optimizer.step()

            if replay_examples_per_class:
                memory_indices.extend(
                    _class_indices(
                        train_data.targets,
                        classes,
                        seed=seed + 5000 + experience_index,
                        maximum_per_class=replay_examples_per_class,
                    )
                )

            row: list[float | None] = [None] * len(experiences)
            task_aware_row: list[float | None] = [None] * len(experiences)
            all_seen_indices = []
            for evaluation_index in range(experience_index + 1):
                evaluation_classes = experiences[evaluation_index]
                test_indices = _class_indices(
                    test_data.targets,
                    evaluation_classes,
                    seed=seed,
                )
                test_loader = _loader(
                    Subset(test_data, test_indices),
                    batch_size,
                    seed=seed,
                    shuffle=False,
                )
                row[evaluation_index], _ = _evaluate(model, test_loader, device)
                task_aware_row[evaluation_index], _ = _evaluate(
                    model,
                    test_loader,
                    device,
                    allowed_classes=evaluation_classes,
                )
                all_seen_indices.extend(test_indices)
            seen_loader = _loader(
                Subset(test_data, all_seen_indices),
                batch_size,
                seed=seed,
                shuffle=False,
            )
            _, representation_norm = _evaluate(model, seen_loader, device)
            representation_norms.append(representation_norm)
            accuracy_matrix.append(row)
            task_aware_matrix.append(task_aware_row)

        average_incremental, final_average, forgetting, acquisition = (
            _summarize_accuracy_matrix(accuracy_matrix)
        )
        _, task_aware_final, task_aware_forgetting, _ = _summarize_accuracy_matrix(
            task_aware_matrix
        )
        results.append(
            ContinualResult(
                mode=mode,
                seed=seed,
                replay_examples_per_class=replay_examples_per_class,
                class_order=class_order,
                experiences=experiences,
                accuracy_matrix=accuracy_matrix,
                task_aware_accuracy_matrix=task_aware_matrix,
                average_incremental_accuracy=average_incremental,
                final_average_accuracy=final_average,
                average_forgetting=forgetting,
                mean_new_experience_accuracy=acquisition,
                task_aware_final_accuracy=task_aware_final,
                task_aware_forgetting=task_aware_forgetting,
                representation_norms=representation_norms,
                parameter_count=trainable_parameter_count(model),
            )
        )
    return results
