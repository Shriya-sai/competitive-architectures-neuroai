"""Signed graph construction for lateral feature interactions."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SignedMasks:
    """Disjoint directed masks using the convention mask[target, source]."""

    cooperative: Tensor
    competitive: Tensor
    groups: Tensor

    def validate(self) -> None:
        """Raise when the masks violate the signed-graph contract."""
        positive = self.cooperative
        negative = self.competitive
        if positive.shape != negative.shape or positive.ndim != 2:
            raise ValueError("signed masks must be square matrices of equal shape")
        if positive.shape[0] != positive.shape[1]:
            raise ValueError("signed masks must be square")
        if self.groups.shape != (positive.shape[0],):
            raise ValueError("one group label is required per node")
        if not torch.all((positive == 0) | (positive == 1)):
            raise ValueError("cooperative mask must be binary")
        if not torch.all((negative == 0) | (negative == 1)):
            raise ValueError("competitive mask must be binary")
        if torch.any(torch.diagonal(positive)) or torch.any(torch.diagonal(negative)):
            raise ValueError("self-interactions are not allowed")
        if torch.any((positive > 0) & (negative > 0)):
            raise ValueError("cooperative and competitive masks must be disjoint")


def structured_signed_masks(
    channels: int,
    groups: int = 4,
    cooperative_degree: int = 2,
    competitive_degree: int = 2,
    seed: int = 0,
) -> SignedMasks:
    """Create regular within-group cooperation and between-group competition."""
    if channels <= 1 or groups <= 1 or channels % groups:
        raise ValueError("channels must be divisible among at least two groups")
    group_size = channels // groups
    if not 0 <= cooperative_degree < group_size:
        raise ValueError("cooperative degree must be smaller than group size")
    if not 0 <= competitive_degree <= channels - group_size:
        raise ValueError("competitive degree exceeds available between-group nodes")

    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(groups), group_size)
    rng.shuffle(labels)
    members = [np.flatnonzero(labels == group) for group in range(groups)]

    positive = np.zeros((channels, channels), dtype=np.float32)
    negative = np.zeros_like(positive)

    for group_members in members:
        for position, source in enumerate(group_members):
            for offset in range(1, cooperative_degree + 1):
                target = group_members[(position + offset) % group_size]
                positive[target, source] = 1

    for group, group_members in enumerate(members):
        for position, source in enumerate(group_members):
            for edge_index in range(competitive_degree):
                group_offset = 1 + edge_index // group_size
                target_group = (group + group_offset) % groups
                target_position = (position + edge_index % group_size) % group_size
                target = members[target_group][target_position]
                negative[target, source] = 1

    masks = SignedMasks(
        cooperative=torch.from_numpy(positive),
        competitive=torch.from_numpy(negative),
        groups=torch.from_numpy(labels.astype(np.int64)),
    )
    masks.validate()
    return masks


def _rewire_mask(
    mask: np.ndarray,
    forbidden: np.ndarray,
    rng: np.random.Generator,
    swaps_per_edge: int,
) -> np.ndarray:
    edges = [tuple(edge) for edge in np.argwhere(mask > 0)]
    edge_set = set(edges)
    required = swaps_per_edge * len(edges)
    accepted = 0
    attempts = 0
    max_attempts = max(100, required * 100)

    while accepted < required and attempts < max_attempts:
        attempts += 1
        first, second = rng.choice(len(edges), size=2, replace=False)
        target_a, source_a = edges[first]
        target_b, source_b = edges[second]
        proposed_a = (target_a, source_b)
        proposed_b = (target_b, source_a)
        old_a = (target_a, source_a)
        old_b = (target_b, source_b)
        if target_a == source_b or target_b == source_a:
            continue
        if proposed_a == proposed_b:
            continue
        remaining = edge_set - {old_a, old_b}
        if proposed_a in remaining or proposed_b in remaining:
            continue
        if forbidden[proposed_a] or forbidden[proposed_b]:
            continue

        edge_set.remove(old_a)
        edge_set.remove(old_b)
        edge_set.add(proposed_a)
        edge_set.add(proposed_b)
        edges[first] = proposed_a
        edges[second] = proposed_b
        accepted += 1

    if accepted < required:
        raise RuntimeError(f"accepted only {accepted}/{required} edge swaps")

    rewired = np.zeros_like(mask)
    for edge in edges:
        rewired[edge] = 1
    return rewired


def rewire_signed_masks(
    masks: SignedMasks,
    seed: int,
    swaps_per_edge: int = 10,
) -> SignedMasks:
    """Randomize placement while preserving every signed in/out degree."""
    masks.validate()
    if swaps_per_edge < 1:
        raise ValueError("swaps_per_edge must be positive")
    rng = np.random.default_rng(seed)
    positive = masks.cooperative.cpu().numpy().astype(bool)
    negative = masks.competitive.cpu().numpy().astype(bool)
    rewired_positive = _rewire_mask(positive, negative, rng, swaps_per_edge)
    rewired_negative = _rewire_mask(
        negative,
        rewired_positive.astype(bool),
        rng,
        swaps_per_edge,
    )
    rewired = SignedMasks(
        cooperative=torch.from_numpy(rewired_positive.astype(np.float32)),
        competitive=torch.from_numpy(rewired_negative.astype(np.float32)),
        groups=masks.groups.clone(),
    )
    rewired.validate()
    return rewired


def signed_degrees(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Return directed in-degree and out-degree for mask[target, source]."""
    return mask.sum(dim=1), mask.sum(dim=0)
