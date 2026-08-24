"""Controlled CIFAR-10 trainability pilot for the core model conditions."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from competitive_architectures.models import paired_models, trainable_parameter_count


@dataclass(frozen=True)
class CifarSmokeResult:
    mode: str
    seed: int
    epochs: int
    train_samples: int
    test_samples: int
    parameter_count: int
    initial_test_accuracy: float
    final_train_loss: float
    final_test_accuracy: float
    mean_representation_norm: float

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _subset(dataset: Dataset, count: int, seed: int) -> Subset:
    if not 0 < count <= len(dataset):
        raise ValueError("subset count must be within the dataset")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def _loader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    norm_sum = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        representations = model.representations(images)
        predictions = model.classifier(representations).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
        norm_sum += float(representations.norm(dim=1).sum())
    return correct / total, norm_sum / total


def run_cifar10_smoke(
    data_dir: str | Path,
    seed: int = 23,
    epochs: int = 3,
    train_samples: int = 6000,
    test_samples: int = 2000,
    batch_size: int = 128,
) -> list[CifarSmokeResult]:
    """Train paired core models on a fixed CIFAR-10 subset without augmentation."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    root = Path(data_dir)
    train_data = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    train_subset = _subset(train_data, train_samples, seed)
    test_subset = _subset(test_data, test_samples, seed + 1)
    device = _device()
    results = []

    for mode, model in paired_models(seed).items():
        model = model.to(device)
        train_loader = _loader(train_subset, batch_size, seed + 2, shuffle=True)
        test_loader = _loader(test_subset, batch_size, seed + 3, shuffle=False)
        initial_accuracy, _ = _evaluate(model, test_loader, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        final_train_loss = float("nan")
        for _ in range(epochs):
            model.train()
            loss_sum = 0.0
            examples = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * labels.numel()
                examples += labels.numel()
            final_train_loss = loss_sum / examples
        final_accuracy, representation_norm = _evaluate(model, test_loader, device)
        results.append(
            CifarSmokeResult(
                mode=mode,
                seed=seed,
                epochs=epochs,
                train_samples=train_samples,
                test_samples=test_samples,
                parameter_count=trainable_parameter_count(model),
                initial_test_accuracy=initial_accuracy,
                final_train_loss=final_train_loss,
                final_test_accuracy=final_accuracy,
                mean_representation_norm=representation_norm,
            )
        )
    return results
