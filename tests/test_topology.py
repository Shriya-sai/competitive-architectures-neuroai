import torch

from competitive_architectures.graphs import structured_signed_masks
from competitive_architectures.topology import topology_alignment


def test_alignment_recovers_known_group_tuning() -> None:
    masks = structured_signed_masks(
        channels=16,
        groups=4,
        cooperative_degree=3,
        competitive_degree=4,
        seed=5,
    )
    labels = torch.arange(4).repeat_interleave(100)
    generator = torch.Generator().manual_seed(9)
    features = torch.zeros(labels.numel(), 16)
    for channel, group in enumerate(masks.groups):
        features[:, channel] = (labels == group).float()
    features += 0.05 * torch.randn(features.shape, generator=generator)

    result = topology_alignment(
        features,
        labels,
        masks.cooperative,
        masks.competitive,
        masks.groups,
    )
    assert result.signed_edge_tuning_gap > 0.8
    assert result.group_tuning_gap > 0.8


def test_alignment_is_near_zero_without_group_organization() -> None:
    masks = structured_signed_masks(16, groups=4, seed=7)
    labels = torch.arange(4).repeat_interleave(200)
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(labels.numel(), 16, generator=generator)
    result = topology_alignment(
        features,
        labels,
        masks.cooperative,
        masks.competitive,
        masks.groups,
    )
    assert abs(result.signed_edge_tuning_gap) < 0.3
    assert abs(result.group_tuning_gap) < 0.3
