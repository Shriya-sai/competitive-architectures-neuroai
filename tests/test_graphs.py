import torch

from competitive_architectures.graphs import (
    edge_overlap_fraction,
    rewire_signed_masks,
    signed_degrees,
    structured_signed_masks,
)


def test_structured_masks_encode_within_and_between_group_edges() -> None:
    masks = structured_signed_masks(
        channels=16,
        groups=4,
        cooperative_degree=2,
        competitive_degree=2,
        seed=3,
    )
    target, source = torch.where(masks.cooperative > 0)
    assert torch.all(masks.groups[target] == masks.groups[source])
    target, source = torch.where(masks.competitive > 0)
    assert torch.all(masks.groups[target] != masks.groups[source])


def test_rewiring_preserves_every_signed_degree() -> None:
    structured = structured_signed_masks(24, groups=4, seed=4)
    rewired = rewire_signed_masks(structured, seed=9)
    for original, randomized in (
        (structured.cooperative, rewired.cooperative),
        (structured.competitive, rewired.competitive),
    ):
        original_in, original_out = signed_degrees(original)
        randomized_in, randomized_out = signed_degrees(randomized)
        assert torch.equal(original_in, randomized_in)
        assert torch.equal(original_out, randomized_out)
    assert not torch.equal(structured.cooperative, rewired.cooperative)
    assert not torch.equal(structured.competitive, rewired.competitive)


def test_graph_generation_is_seed_reproducible() -> None:
    first = structured_signed_masks(16, seed=12)
    second = structured_signed_masks(16, seed=12)
    assert torch.equal(first.groups, second.groups)
    assert torch.equal(first.cooperative, second.cooperative)
    assert torch.equal(first.competitive, second.competitive)


def test_more_rewiring_reduces_edge_overlap_without_changing_degree() -> None:
    structured = structured_signed_masks(24, groups=4, seed=5)
    light = rewire_signed_masks(structured, seed=6, swaps_per_edge=0.1)
    heavy = rewire_signed_masks(structured, seed=6, swaps_per_edge=10)
    light_overlap = edge_overlap_fraction(structured.cooperative, light.cooperative)
    heavy_overlap = edge_overlap_fraction(structured.cooperative, heavy.cooperative)
    assert light_overlap > heavy_overlap
