"""Small controlled image models for the first real-task experiments."""

from typing import Literal

import torch
from torch import Tensor, nn

from competitive_architectures.graphs import (
    rewire_signed_masks,
    structured_signed_masks,
)
from competitive_architectures.lateral import (
    GatedSignedLateral,
    SignedBottleneck,
    SignedLateral,
)

InteractionMode = Literal["standard", "random_signed", "structured_signed"]
PathwayMode = Literal["weak_residual", "gated_residual", "signed_bottleneck"]


class TinyCifarCNN(nn.Module):
    """A small shared backbone with an optional signed representation layer."""

    representation_width = 64

    def __init__(
        self,
        mode: InteractionMode = "standard",
        classes: int = 10,
        graph_seed: int = 0,
        cooperative_gain: float = 0.25,
        competitive_gain: float = 0.25,
        residual_scale: float = 0.5,
        pathway_mode: PathwayMode = "weak_residual",
    ) -> None:
        super().__init__()
        if mode not in ("standard", "random_signed", "structured_signed"):
            raise ValueError(f"unknown interaction mode: {mode}")
        self.mode = mode
        self.pathway_mode = pathway_mode
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        if mode == "standard":
            self.interaction: nn.Module = nn.Identity()
        else:
            masks = structured_signed_masks(
                self.representation_width,
                groups=4,
                cooperative_degree=4,
                competitive_degree=4,
                seed=graph_seed,
            )
            if mode == "random_signed":
                masks = rewire_signed_masks(masks, seed=graph_seed + 1)
            pathway_class = {
                "weak_residual": SignedLateral,
                "gated_residual": GatedSignedLateral,
                "signed_bottleneck": SignedBottleneck,
            }.get(pathway_mode)
            if pathway_class is None:
                raise ValueError(f"unknown pathway mode: {pathway_mode}")
            self.interaction = pathway_class(
                masks,
                cooperative_gain=cooperative_gain,
                competitive_gain=competitive_gain,
                residual_scale=residual_scale,
                initial_magnitude=0.05,
            )
        self.classifier = nn.Linear(self.representation_width, classes)

    def representations(self, images: Tensor) -> Tensor:
        """Return the post-interaction representation used by the classifier."""
        features = self.backbone(images).flatten(1)
        return self.interaction(features)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.representations(images))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def paired_models(
    seed: int,
    classes: int = 10,
    pathway_mode: PathwayMode = "weak_residual",
) -> dict[InteractionMode, TinyCifarCNN]:
    """Construct all core conditions with paired backbone initialization."""
    models = {}
    for mode in ("standard", "random_signed", "structured_signed"):
        torch.manual_seed(seed)
        models[mode] = TinyCifarCNN(
            mode=mode,
            classes=classes,
            graph_seed=seed,
            pathway_mode=pathway_mode,
        )
    return models
