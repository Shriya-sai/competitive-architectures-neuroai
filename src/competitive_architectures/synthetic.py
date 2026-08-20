"""Synthetic construct-validity experiment for signed lateral interactions."""

from dataclasses import asdict, dataclass
from statistics import mean

import torch
from torch import Tensor

from competitive_architectures.graphs import (
    SignedMasks,
    rewire_signed_masks,
    structured_signed_masks,
)
from competitive_architectures.lateral import SignedLateral


@dataclass(frozen=True)
class SyntheticLearningResult:
    initial_structured_loss: float
    final_structured_loss: float
    initial_rewired_loss: float
    final_rewired_loss: float
    structured_parameter_count: int
    rewired_parameter_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class StressConditionResult:
    teacher_magnitude: float
    target_noise: float
    samples: int
    seeds: int
    mean_structured_loss: float
    mean_rewired_loss: float
    mean_loss_advantage: float
    structured_wins: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _fit_layer(
    masks: SignedMasks,
    train_features: Tensor,
    train_targets: Tensor,
    test_features: Tensor,
    test_targets: Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[float, float, int]:
    layer = SignedLateral(
        masks,
        cooperative_gain=1.0,
        competitive_gain=1.0,
        initial_magnitude=0.01,
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        initial_loss = float(loss_fn(layer(test_features), test_targets))
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(layer(train_features), train_targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_fn(layer(test_features), test_targets))
    parameters = sum(parameter.numel() for parameter in layer.parameters())
    return initial_loss, final_loss, parameters


def run_synthetic_learning_experiment(
    seed: int = 17,
    channels: int = 16,
    samples: int = 1024,
    steps: int = 300,
    learning_rate: float = 0.05,
    teacher_magnitude: float = 0.2,
    target_noise: float = 0.0,
) -> SyntheticLearningResult:
    """Recover a known signed mapping with correct and rewired topologies."""
    generator = torch.Generator().manual_seed(seed)
    structured = structured_signed_masks(
        channels,
        groups=4,
        cooperative_degree=2,
        competitive_degree=2,
        seed=seed,
    )
    rewired = rewire_signed_masks(structured, seed=seed + 1)
    teacher = SignedLateral(
        structured,
        cooperative_gain=float(teacher_magnitude > 0),
        competitive_gain=float(teacher_magnitude > 0),
        initial_magnitude=max(teacher_magnitude, 0.01),
    )
    teacher.requires_grad_(False)

    features = torch.randn(samples, channels, generator=generator)
    split = samples * 3 // 4
    with torch.no_grad():
        targets = teacher(features)
        if target_noise:
            targets = targets + target_noise * torch.randn(
                targets.shape,
                generator=generator,
            )
    train_features, test_features = features[:split], features[split:]
    train_targets, test_targets = targets[:split], targets[split:]

    torch.manual_seed(seed + 2)
    structured_result = _fit_layer(
        structured,
        train_features,
        train_targets,
        test_features,
        test_targets,
        steps,
        learning_rate,
    )
    torch.manual_seed(seed + 2)
    rewired_result = _fit_layer(
        rewired,
        train_features,
        train_targets,
        test_features,
        test_targets,
        steps,
        learning_rate,
    )
    return SyntheticLearningResult(
        initial_structured_loss=structured_result[0],
        final_structured_loss=structured_result[1],
        initial_rewired_loss=rewired_result[0],
        final_rewired_loss=rewired_result[1],
        structured_parameter_count=structured_result[2],
        rewired_parameter_count=rewired_result[2],
    )


def run_synthetic_stress_suite(
    seeds: tuple[int, ...] = (3, 11, 19, 27, 35),
    teacher_magnitudes: tuple[float, ...] = (0.0, 0.05, 0.2),
    target_noise_levels: tuple[float, ...] = (0.0, 0.1, 0.5),
    sample_sizes: tuple[int, ...] = (128, 512),
    steps: int = 200,
) -> list[StressConditionResult]:
    """Evaluate recovery across signal, noise, sample size, and graph seeds."""
    summaries = []
    for magnitude in teacher_magnitudes:
        for noise in target_noise_levels:
            for samples in sample_sizes:
                runs = [
                    run_synthetic_learning_experiment(
                        seed=seed,
                        samples=samples,
                        steps=steps,
                        teacher_magnitude=magnitude,
                        target_noise=noise,
                    )
                    for seed in seeds
                ]
                structured_losses = [run.final_structured_loss for run in runs]
                rewired_losses = [run.final_rewired_loss for run in runs]
                advantages = [
                    rewired - structured
                    for structured, rewired in zip(
                        structured_losses,
                        rewired_losses,
                        strict=True,
                    )
                ]
                summaries.append(
                    StressConditionResult(
                        teacher_magnitude=magnitude,
                        target_noise=noise,
                        samples=samples,
                        seeds=len(seeds),
                        mean_structured_loss=mean(structured_losses),
                        mean_rewired_loss=mean(rewired_losses),
                        mean_loss_advantage=mean(advantages),
                        structured_wins=sum(advantage > 0 for advantage in advantages),
                    )
                )
    return summaries
