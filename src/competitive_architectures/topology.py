"""Measurements of functional alignment with a signed channel topology."""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TopologyAlignment:
    cooperative_tuning_similarity: float
    competitive_tuning_similarity: float
    signed_edge_tuning_gap: float
    within_group_tuning_similarity: float
    between_group_tuning_similarity: float
    group_tuning_gap: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def topology_alignment(
    features: Tensor,
    labels: Tensor,
    cooperative_mask: Tensor,
    competitive_mask: Tensor,
    groups: Tensor,
) -> TopologyAlignment:
    """Compare class-tuning similarity along signed edges and structural groups."""
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("features must be 2D and labels must be 1D")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must contain equal observations")
    channels = features.shape[1]
    if cooperative_mask.shape != (channels, channels):
        raise ValueError("cooperative mask does not match feature width")
    if competitive_mask.shape != (channels, channels):
        raise ValueError("competitive mask does not match feature width")
    if groups.shape != (channels,):
        raise ValueError("groups do not match feature width")

    classes = torch.unique(labels, sorted=True)
    tuning = torch.stack([features[labels == label].mean(dim=0) for label in classes])
    tuning = tuning.T
    tuning = tuning - tuning.mean(dim=1, keepdim=True)
    tuning = tuning / tuning.norm(dim=1, keepdim=True).clamp_min(1e-12)
    similarity = tuning @ tuning.T

    cooperative = cooperative_mask.to(similarity.device).bool()
    competitive = competitive_mask.to(similarity.device).bool()
    group_labels = groups.to(similarity.device)
    diagonal = torch.eye(channels, dtype=torch.bool, device=similarity.device)
    within = (group_labels[:, None] == group_labels[None, :]) & ~diagonal
    between = group_labels[:, None] != group_labels[None, :]

    cooperative_mean = similarity[cooperative].mean()
    competitive_mean = similarity[competitive].mean()
    within_mean = similarity[within].mean()
    between_mean = similarity[between].mean()
    return TopologyAlignment(
        cooperative_tuning_similarity=float(cooperative_mean.cpu()),
        competitive_tuning_similarity=float(competitive_mean.cpu()),
        signed_edge_tuning_gap=float((cooperative_mean - competitive_mean).cpu()),
        within_group_tuning_similarity=float(within_mean.cpu()),
        between_group_tuning_similarity=float(between_mean.cpu()),
        group_tuning_gap=float((within_mean - between_mean).cpu()),
    )
