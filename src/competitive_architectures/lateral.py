"""PyTorch module for explicit cooperative and competitive interactions."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from competitive_architectures.graphs import SignedMasks


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("initial magnitude must be positive")
    return math.log(math.expm1(value))


class SignedLateral(nn.Module):
    """Residual signed interaction over the final tensor dimension."""

    def __init__(
        self,
        masks: SignedMasks,
        cooperative_gain: float = 0.0,
        competitive_gain: float = 0.0,
        residual_scale: float = 1.0,
        initial_magnitude: float = 0.05,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        masks.validate()
        if cooperative_gain < 0 or competitive_gain < 0 or residual_scale < 0:
            raise ValueError("gains and residual scale must be nonnegative")
        self.register_buffer("cooperative_mask", masks.cooperative.clone())
        self.register_buffer("competitive_mask", masks.competitive.clone())
        cooperative_targets, cooperative_sources = torch.where(masks.cooperative > 0)
        competitive_targets, competitive_sources = torch.where(masks.competitive > 0)
        self.register_buffer("cooperative_targets", cooperative_targets)
        self.register_buffer("cooperative_sources", cooperative_sources)
        self.register_buffer("competitive_targets", competitive_targets)
        self.register_buffer("competitive_sources", competitive_sources)
        initial_raw = _inverse_softplus(initial_magnitude)
        self.cooperative_raw = nn.Parameter(
            torch.full((cooperative_targets.numel(),), initial_raw)
        )
        self.competitive_raw = nn.Parameter(
            torch.full((competitive_targets.numel(),), initial_raw)
        )
        self.cooperative_gain = float(cooperative_gain)
        self.competitive_gain = float(competitive_gain)
        self.residual_scale = float(residual_scale)
        self.activation = activation if activation is not None else nn.Identity()

    @property
    def cooperative_weights(self) -> Tensor:
        weights = torch.zeros_like(self.cooperative_mask)
        return weights.index_put(
            (self.cooperative_targets, self.cooperative_sources),
            F.softplus(self.cooperative_raw),
        )

    @property
    def competitive_weights(self) -> Tensor:
        weights = torch.zeros_like(self.competitive_mask)
        return weights.index_put(
            (self.competitive_targets, self.competitive_sources),
            F.softplus(self.competitive_raw),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.cooperative_mask.shape[0]:
            raise ValueError("final feature dimension does not match graph width")
        cooperative = F.linear(features, self.cooperative_weights)
        competitive = F.linear(features, self.competitive_weights)
        interaction = (
            self.cooperative_gain * cooperative - self.competitive_gain * competitive
        )
        return self.activation(features + self.residual_scale * interaction)
