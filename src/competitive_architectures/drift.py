"""Experience-wise representation and readout drift diagnostics."""

import json
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms

from competitive_architectures.continual import _class_indices, _class_order, _loader
from competitive_architectures.development import DEVELOPMENT_SEEDS
from competitive_architectures.models import paired_models


@dataclass
class RepresentationSnapshot:
    centroids: dict[int, Tensor]
    dispersions: dict[int, float]
    margins: dict[int, float]
    accuracies: dict[int, float]
    group_profiles: dict[int, Tensor]
    classifier_rows: Tensor


def _cosine_distance(first: Tensor, second: Tensor) -> float:
    return float((1 - F.cosine_similarity(first[None], second[None])).item())


def _normalized_group_profiles(representations: Tensor, groups: Tensor) -> Tensor:
    profiles = torch.stack(
        [
            representations[:, groups == group].abs().mean(dim=1)
            for group in torch.unique(groups)
        ],
        dim=1,
    )
    return profiles / profiles.sum(dim=1, keepdim=True).clamp_min(1e-12)


def compare_snapshots(
    previous: RepresentationSnapshot,
    current: RepresentationSnapshot,
    old_classes: list[int],
    new_classes: list[int],
) -> dict[str, float]:
    """Decompose one post-experience change using fixed held-out classes."""
    centroid_drift = [
        _cosine_distance(previous.centroids[label], current.centroids[label])
        for label in old_classes
    ]
    group_drift = [
        _cosine_distance(previous.group_profiles[label], current.group_profiles[label])
        for label in old_classes
    ]
    classifier_drift = [
        _cosine_distance(
            previous.classifier_rows[label], current.classifier_rows[label]
        )
        for label in old_classes
    ]
    old_new_similarity = []
    for old_label in old_classes:
        old_centroid = current.centroids[old_label]
        similarities = [
            float(
                F.cosine_similarity(
                    old_centroid[None], current.centroids[new_label][None]
                ).item()
            )
            for new_label in new_classes
        ]
        old_new_similarity.append(max(similarities))
    return {
        "old_centroid_cosine_drift": float(np.mean(centroid_drift)),
        "old_group_profile_cosine_drift": float(np.mean(group_drift)),
        "old_classifier_cosine_drift": float(np.mean(classifier_drift)),
        "old_dispersion_change": float(
            np.mean(
                [
                    current.dispersions[label] - previous.dispersions[label]
                    for label in old_classes
                ]
            )
        ),
        "old_margin_change": float(
            np.mean(
                [
                    current.margins[label] - previous.margins[label]
                    for label in old_classes
                ]
            )
        ),
        "old_accuracy_change": float(
            np.mean(
                [
                    current.accuracies[label] - previous.accuracies[label]
                    for label in old_classes
                ]
            )
        ),
        "old_new_centroid_max_similarity": float(np.mean(old_new_similarity)),
    }


@torch.no_grad()
def _snapshot(model: torch.nn.Module, test_data: datasets.CIFAR10, seed: int) -> RepresentationSnapshot:
    device = next(model.parameters()).device
    model.eval()
    centroids, dispersions, margins, accuracies, group_profiles = {}, {}, {}, {}, {}
    groups = model.interaction_groups
    for label in range(10):
        indices = _class_indices(test_data.targets, [label], seed=seed)
        loader = _loader(Subset(test_data, indices), 128, seed, False)
        representations, logits_all = [], []
        for images, _ in loader:
            representation = model.representations(images.to(device))
            representations.append(representation.cpu())
            logits_all.append(model.classifier(representation).cpu())
        representation = torch.cat(representations)
        logits = torch.cat(logits_all)
        centroid = representation.mean(dim=0)
        centroids[label] = centroid
        dispersions[label] = float((representation - centroid).square().sum(dim=1).mean())
        competing = logits.clone()
        competing[:, label] = float("-inf")
        margins[label] = float((logits[:, label] - competing.max(dim=1).values).mean())
        accuracies[label] = float((logits.argmax(dim=1) == label).float().mean())
        group_profiles[label] = torch.stack(
            [representation[:, groups.cpu() == group].abs().mean() for group in torch.unique(groups.cpu())]
        )
    return RepresentationSnapshot(
        centroids=centroids,
        dispersions=dispersions,
        margins=margins,
        accuracies=accuracies,
        group_profiles=group_profiles,
        classifier_rows=model.classifier.weight.detach().cpu().clone(),
    )


@torch.no_grad()
def _prototype(
    model: torch.nn.Module,
    train_data: datasets.CIFAR10,
    indices: list[int],
    seed: int,
    label: int,
) -> tuple[Tensor, float]:
    device = next(model.parameters()).device
    profiles = []
    correct = 0
    total = 0
    model.eval()
    for images, _ in _loader(Subset(train_data, indices), 128, seed, False):
        representations = model.representations(images.to(device))
        predictions = model.classifier(representations).argmax(dim=1)
        correct += int((predictions == label).sum())
        total += predictions.numel()
        profiles.append(
            _normalized_group_profiles(representations, model.interaction_groups)
        )
    return torch.cat(profiles).mean(dim=0).detach(), correct / total


def _train_seed(
    data_dir: str | Path,
    seed: int,
    profile_stability_weight: float = 0.0,
    selective_consolidation: bool = False,
) -> list[dict[str, object]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ]
    )
    train_data = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform)
    order = _class_order(seed)
    experiences = [order[index : index + 2] for index in range(0, 10, 2)]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = paired_models(seed, pathway_mode="signed_bottleneck")
    runs = []
    for mode in ("random_signed", "structured_signed"):
        model = models[mode].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        memory_indices: list[int] = []
        frozen_profiles: dict[int, Tensor] = {}
        consolidation_scores: dict[int, float] = {}
        previous = None
        transitions = []
        acquisition_accuracies = []
        for experience_index, classes in enumerate(experiences):
            current_indices = _class_indices(
                train_data.targets,
                classes,
                seed + 100 * experience_index,
                maximum_per_class=2000,
            )
            for epoch in range(3):
                current_loader = _loader(
                    Subset(train_data, current_indices),
                    64,
                    seed + 1000 * experience_index + epoch,
                    True,
                )
                replay_batches = []
                if memory_indices:
                    replay_batches = list(
                        _loader(
                            Subset(train_data, memory_indices),
                            64,
                            seed + 10000 + 1000 * experience_index + epoch,
                            True,
                        )
                    )
                replay_iterator = cycle(replay_batches) if replay_batches else None
                model.train()
                for images, labels in current_loader:
                    replay_count = 0
                    if replay_iterator is not None:
                        replay_images, replay_labels = next(replay_iterator)
                        replay_count = replay_labels.numel()
                        images = torch.cat((images, replay_images))
                        labels = torch.cat((labels, replay_labels))
                    optimizer.zero_grad()
                    images = images.to(device)
                    labels = labels.to(device)
                    representations = model.representations(images)
                    loss = F.cross_entropy(model.classifier(representations), labels)
                    if profile_stability_weight and replay_count:
                        replay_profiles = _normalized_group_profiles(
                            representations[-replay_count:], model.interaction_groups
                        )
                        targets = torch.stack(
                            [
                                frozen_profiles[label]
                                for label in labels[-replay_count:].cpu().tolist()
                            ]
                        )
                        per_sample_penalty = (replay_profiles - targets).square().mean(
                            dim=1
                        )
                        if selective_consolidation:
                            sample_weights = torch.tensor(
                                [
                                    consolidation_scores[label]
                                    for label in labels[-replay_count:].cpu().tolist()
                                ],
                                device=device,
                            )
                            per_sample_penalty = per_sample_penalty * sample_weights
                        loss = loss + profile_stability_weight * per_sample_penalty.mean()
                    loss.backward()
                    optimizer.step()
            new_memory_by_class = {
                label: _class_indices(
                    train_data.targets,
                    [label],
                    seed + 5000 + experience_index,
                    maximum_per_class=200,
                )
                for label in classes
            }
            for label, indices in new_memory_by_class.items():
                memory_indices.extend(indices)
                prototype, score = _prototype(
                    model, train_data, indices, seed, label
                )
                frozen_profiles[label] = prototype
                consolidation_scores[label] = score
            current = _snapshot(model, test_data, seed)
            acquisition_accuracies.append(
                float(np.mean([current.accuracies[label] for label in classes]))
            )
            if previous is not None:
                transition = compare_snapshots(
                    previous,
                    current,
                    [label for prior in experiences[:experience_index] for label in prior],
                    classes,
                )
                transition["after_experience"] = experience_index + 1
                transitions.append(transition)
            previous = current
        runs.append(
            {
                "seed": seed,
                "mode": mode,
                "profile_stability_weight": profile_stability_weight,
                "selective_consolidation": selective_consolidation,
                "transitions": transitions,
                "mean_new_experience_accuracy": float(
                    np.mean(acquisition_accuracies)
                ),
                "final_average_accuracy": float(
                    np.mean(list(previous.accuracies.values()))
                ),
            }
        )
    return runs


def _summary(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = tuple(runs[0]["transitions"][0].keys())
    metrics = tuple(metric for metric in metrics if metric != "after_experience")
    run_means = {}
    for run in runs:
        run_means[(run["seed"], run["mode"])] = {
            metric: float(np.mean([item[metric] for item in run["transitions"]]))
            for metric in metrics
        }
    condition_means = {
        mode: {
            metric: float(
                np.mean([run_means[(seed, mode)][metric] for seed in DEVELOPMENT_SEEDS])
            )
            for metric in metrics
        }
        for mode in ("random_signed", "structured_signed")
    }
    contrasts = {
        metric: {
            "paired_differences": [
                run_means[(seed, "structured_signed")][metric]
                - run_means[(seed, "random_signed")][metric]
                for seed in DEVELOPMENT_SEEDS
            ],
        }
        for metric in metrics
    }
    for value in contrasts.values():
        value["mean_difference"] = float(np.mean(value["paired_differences"]))
    return {"condition_means": condition_means, "structured_minus_random": contrasts}


def run_drift_screen(data_dir: str | Path, output_path: str | Path) -> dict[str, object]:
    """Run and checkpoint the exposed five-seed drift decomposition."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": "exploratory_exposed_seed_representational_drift_screen",
        "seeds": list(DEVELOPMENT_SEEDS),
        "completed_seeds": [],
        "runs": [],
    }
    for seed in DEVELOPMENT_SEEDS:
        print(f"Starting drift seed {seed}", flush=True)
        payload["runs"].extend(_train_seed(data_dir, seed))
        payload["completed_seeds"].append(seed)
        destination.write_text(json.dumps(payload, indent=2))
        print(f"Completed drift seed {seed}", flush=True)
    payload["summary"] = _summary(payload["runs"])
    destination.write_text(json.dumps(payload, indent=2))
    return payload


def run_profile_stability_calibration(
    data_dir: str | Path,
    output_path: str | Path,
    weights: tuple[float, ...] = (0.0, 1.0, 10.0),
) -> dict[str, object]:
    """Calibrate intervention strength on the original development seed only."""
    payload = {
        "status": "exploratory_profile_stability_calibration",
        "seed": 23,
        "weights": list(weights),
        "runs": [],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for weight in weights:
        print(f"Starting profile-stability weight {weight:g}", flush=True)
        payload["runs"].extend(_train_seed(data_dir, 23, weight))
        destination.write_text(json.dumps(payload, indent=2))
        print(f"Completed profile-stability weight {weight:g}", flush=True)
    return payload


def _intervention_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "mean_group_drift",
        "mean_centroid_drift",
        "mean_margin_change",
        "mean_old_accuracy_change",
        "mean_new_experience_accuracy",
        "final_average_accuracy",
    )
    flattened = {}
    for run in runs:
        transitions = run["transitions"]
        flattened[(run["seed"], run["mode"], run["profile_stability_weight"])] = {
            "mean_group_drift": float(
                np.mean([item["old_group_profile_cosine_drift"] for item in transitions])
            ),
            "mean_centroid_drift": float(
                np.mean([item["old_centroid_cosine_drift"] for item in transitions])
            ),
            "mean_margin_change": float(
                np.mean([item["old_margin_change"] for item in transitions])
            ),
            "mean_old_accuracy_change": float(
                np.mean([item["old_accuracy_change"] for item in transitions])
            ),
            "mean_new_experience_accuracy": run["mean_new_experience_accuracy"],
            "final_average_accuracy": run["final_average_accuracy"],
        }
    effects = {}
    for mode in ("random_signed", "structured_signed"):
        effects[mode] = {}
        for metric in metrics:
            differences = [
                flattened[(seed, mode, 10.0)][metric]
                - flattened[(seed, mode, 0.0)][metric]
                for seed in DEVELOPMENT_SEEDS
            ]
            effects[mode][metric] = {
                "paired_differences": differences,
                "mean_difference": float(np.mean(differences)),
            }
    return {"regularized_minus_control": effects}


def run_profile_stability_replication(
    data_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Replicate frozen weight 10 against control on five exposed seeds."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": "exploratory_profile_stability_causal_replication",
        "selection_rule": "drift reduction with acquisition preservation",
        "selected_weight": 10.0,
        "seeds": list(DEVELOPMENT_SEEDS),
        "completed_seeds": [],
        "runs": [],
    }
    for seed in DEVELOPMENT_SEEDS:
        print(f"Starting stability replication seed {seed}", flush=True)
        for weight in (0.0, 10.0):
            payload["runs"].extend(_train_seed(data_dir, seed, weight))
        payload["completed_seeds"].append(seed)
        destination.write_text(json.dumps(payload, indent=2))
        print(f"Completed stability replication seed {seed}", flush=True)
    payload["summary"] = _intervention_summary(payload["runs"])
    destination.write_text(json.dumps(payload, indent=2))
    return payload


def _selective_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "mean_group_drift",
        "mean_old_accuracy_change",
        "mean_new_experience_accuracy",
        "final_average_accuracy",
    )
    flattened = {}
    for run in runs:
        transitions = run["transitions"]
        condition = (
            "control"
            if run["profile_stability_weight"] == 0
            else "selective"
            if run["selective_consolidation"]
            else "global"
        )
        flattened[(run["seed"], run["mode"], condition)] = {
            "mean_group_drift": float(
                np.mean([item["old_group_profile_cosine_drift"] for item in transitions])
            ),
            "mean_old_accuracy_change": float(
                np.mean([item["old_accuracy_change"] for item in transitions])
            ),
            "mean_new_experience_accuracy": run["mean_new_experience_accuracy"],
            "final_average_accuracy": run["final_average_accuracy"],
        }
    output = {}
    for mode in ("random_signed", "structured_signed"):
        output[mode] = {}
        for condition in ("control", "global", "selective"):
            output[mode][condition] = {
                metric: float(
                    np.mean(
                        [
                            flattened[(seed, mode, condition)][metric]
                            for seed in DEVELOPMENT_SEEDS
                        ]
                    )
                )
                for metric in metrics
            }
    return output


def run_selective_consolidation_screen(
    data_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Compare control, global stability and selective consolidation."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": "exploratory_selective_consolidation_screen",
        "profile_stability_weight": 10.0,
        "selection_signal": "replay-memory acquisition accuracy",
        "seeds": list(DEVELOPMENT_SEEDS),
        "completed_seeds": [],
        "runs": [],
    }
    conditions = ((0.0, False), (10.0, False), (10.0, True))
    for seed in DEVELOPMENT_SEEDS:
        print(f"Starting selective-consolidation seed {seed}", flush=True)
        for weight, selective in conditions:
            payload["runs"].extend(_train_seed(data_dir, seed, weight, selective))
        payload["completed_seeds"].append(seed)
        destination.write_text(json.dumps(payload, indent=2))
        print(f"Completed selective-consolidation seed {seed}", flush=True)
    payload["summary"] = _selective_summary(payload["runs"])
    destination.write_text(json.dumps(payload, indent=2))
    return payload
