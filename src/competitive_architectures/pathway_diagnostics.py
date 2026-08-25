"""Causal diagnostics for the signed pathway in the continual-learning model."""

from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from competitive_architectures.continual import _class_indices, _class_order, _loader
from competitive_architectures.lateral import SignedLateral
from competitive_architectures.models import paired_models


@dataclass(frozen=True)
class PathwayDiagnostic:
    mode: str
    normal_accuracy: float
    bypass_accuracy: float
    accuracy_effect: float
    prediction_disagreement: float
    residual_to_backbone_ratio: float
    cooperative_to_backbone_ratio: float
    competitive_to_backbone_ratio: float
    post_backbone_cosine: float
    mean_absolute_logit_effect: float
    cooperative_weight_mean: float
    competitive_weight_mean: float
    cooperative_weight_cv: float
    competitive_weight_cv: float
    cooperative_edge_correlation: float
    competitive_edge_correlation: float
    edge_correlation_separation: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _linear_cka(first: Tensor, second: Tensor) -> float:
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    numerator = (first.T @ second).square().sum()
    denominator = torch.sqrt(
        (first.T @ first).square().sum() * (second.T @ second).square().sum()
    )
    return float((numerator / denominator.clamp_min(1e-12)).cpu())


def _coefficient_of_variation(values: Tensor) -> float:
    ratio = values.std(unbiased=False) / values.mean().clamp_min(1e-12)
    return float(ratio.cpu())


@torch.no_grad()
def _diagnose_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[PathwayDiagnostic, Tensor, Tensor]:
    model.eval()
    backbones, posts = [], []
    labels_all, normal_predictions, bypass_predictions = [], [], []
    residual_ratios, cooperative_ratios, competitive_ratios = [], [], []
    cosines, logit_effects = [], []
    interaction = model.interaction
    if not isinstance(interaction, SignedLateral):
        raise TypeError("pathway diagnostics require a SignedLateral layer")

    for images, labels in loader:
        images = images.to(device)
        features = model.backbone(images).flatten(1)
        cooperative = interaction.cooperative_gain * F.linear(
            features, interaction.cooperative_weights
        )
        competitive = interaction.competitive_gain * F.linear(
            features, interaction.competitive_weights
        )
        post = interaction(features)
        normal_logits = model.classifier(post)
        bypass_logits = model.classifier(features)
        denominator = features.norm(dim=1).clamp_min(1e-12)
        backbones.append(features.cpu())
        posts.append(post.cpu())
        labels_all.append(labels)
        normal_predictions.append(normal_logits.argmax(dim=1).cpu())
        bypass_predictions.append(bypass_logits.argmax(dim=1).cpu())
        residual_ratios.append(((post - features).norm(dim=1) / denominator).cpu())
        cooperative_ratios.append((cooperative.norm(dim=1) / denominator).cpu())
        competitive_ratios.append((competitive.norm(dim=1) / denominator).cpu())
        cosines.append(F.cosine_similarity(features, post, dim=1).cpu())
        logit_effects.append((normal_logits - bypass_logits).abs().mean(dim=1).cpu())

    backbone = torch.cat(backbones)
    post = torch.cat(posts)
    labels = torch.cat(labels_all)
    normal = torch.cat(normal_predictions)
    bypass = torch.cat(bypass_predictions)
    correlations = torch.corrcoef(backbone.T)
    cooperative_edge_values = correlations[interaction.cooperative_mask.cpu().bool()]
    competitive_edge_values = correlations[interaction.competitive_mask.cpu().bool()]
    cooperative_weights = interaction.cooperative_weights[
        interaction.cooperative_mask.bool()
    ]
    competitive_weights = interaction.competitive_weights[
        interaction.competitive_mask.bool()
    ]
    normal_accuracy = float((normal == labels).float().mean())
    bypass_accuracy = float((bypass == labels).float().mean())
    diagnostic = PathwayDiagnostic(
        mode=model.mode,
        normal_accuracy=normal_accuracy,
        bypass_accuracy=bypass_accuracy,
        accuracy_effect=normal_accuracy - bypass_accuracy,
        prediction_disagreement=float((normal != bypass).float().mean()),
        residual_to_backbone_ratio=float(torch.cat(residual_ratios).mean()),
        cooperative_to_backbone_ratio=float(torch.cat(cooperative_ratios).mean()),
        competitive_to_backbone_ratio=float(torch.cat(competitive_ratios).mean()),
        post_backbone_cosine=float(torch.cat(cosines).mean()),
        mean_absolute_logit_effect=float(torch.cat(logit_effects).mean()),
        cooperative_weight_mean=float(cooperative_weights.mean().cpu()),
        competitive_weight_mean=float(competitive_weights.mean().cpu()),
        cooperative_weight_cv=_coefficient_of_variation(cooperative_weights),
        competitive_weight_cv=_coefficient_of_variation(competitive_weights),
        cooperative_edge_correlation=float(cooperative_edge_values.mean()),
        competitive_edge_correlation=float(competitive_edge_values.mean()),
        edge_correlation_separation=float(
            cooperative_edge_values.mean() - competitive_edge_values.mean()
        ),
    )
    return diagnostic, backbone, post


def run_pathway_diagnosis(
    data_dir: str | Path,
    seed: int = 31,
    epochs_per_experience: int = 3,
    train_examples_per_class: int = 2000,
    batch_size: int = 128,
    replay_examples_per_class: int = 200,
) -> dict[str, object]:
    """Retrain one frozen paired seed and causally ablate the signed pathway."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ]
    )
    train_data = datasets.CIFAR10(
        data_dir, train=True, download=False, transform=transform
    )
    test_data = datasets.CIFAR10(
        data_dir, train=False, download=False, transform=transform
    )
    order = _class_order(seed)
    experiences = [order[index : index + 2] for index in range(0, 10, 2)]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = paired_models(seed)

    for model in models.values():
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        memory_indices: list[int] = []
        for experience_index, classes in enumerate(experiences):
            current_indices = _class_indices(
                train_data.targets,
                classes,
                seed=seed + 100 * experience_index,
                maximum_per_class=train_examples_per_class,
            )
            for epoch in range(epochs_per_experience):
                current_loader = _loader(
                    Subset(train_data, current_indices),
                    batch_size // 2,
                    seed + 1000 * experience_index + epoch,
                    True,
                )
                replay_batches = []
                if memory_indices:
                    replay_batches = list(
                        _loader(
                            Subset(train_data, memory_indices),
                            batch_size // 2,
                            seed + 10000 + 1000 * experience_index + epoch,
                            True,
                        )
                    )
                replay_iterator = cycle(replay_batches) if replay_batches else None
                model.train()
                for images, labels in current_loader:
                    if replay_iterator is not None:
                        replay_images, replay_labels = next(replay_iterator)
                        images = torch.cat((images, replay_images))
                        labels = torch.cat((labels, replay_labels))
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
            memory_indices.extend(
                _class_indices(
                    train_data.targets,
                    classes,
                    seed=seed + 5000 + experience_index,
                    maximum_per_class=replay_examples_per_class,
                )
            )

    test_loader = _loader(
        Subset(test_data, list(range(len(test_data)))), batch_size, seed, False
    )
    diagnostics = []
    representations: dict[str, dict[str, Tensor]] = {}
    for mode in ("random_signed", "structured_signed"):
        diagnostic, backbone, post = _diagnose_model(models[mode], test_loader, device)
        diagnostics.append(diagnostic.to_dict())
        representations[mode] = {"backbone": backbone, "post": post}
    return {
        "status": "exploratory_single_seed_pathway_diagnosis",
        "seed": seed,
        "configuration": {
            "epochs_per_experience": epochs_per_experience,
            "train_examples_per_class": train_examples_per_class,
            "batch_size": batch_size,
            "replay_examples_per_class": replay_examples_per_class,
        },
        "diagnostics": diagnostics,
        "random_structured_cka": {
            "backbone": _linear_cka(
                representations["random_signed"]["backbone"],
                representations["structured_signed"]["backbone"],
            ),
            "post_interaction": _linear_cka(
                representations["random_signed"]["post"],
                representations["structured_signed"]["post"],
            ),
        },
    }
