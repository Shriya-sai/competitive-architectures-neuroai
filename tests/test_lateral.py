import torch

from competitive_architectures.graphs import SignedMasks, structured_signed_masks
from competitive_architectures.lateral import SignedLateral


def test_zero_gains_recover_input_and_input_gradient() -> None:
    masks = structured_signed_masks(8, cooperative_degree=1, competitive_degree=1)
    layer = SignedLateral(masks)
    features = torch.randn(5, 8, requires_grad=True)
    output = layer(features)
    assert torch.equal(output, features)
    output.sum().backward()
    assert torch.equal(features.grad, torch.ones_like(features))


def test_cooperation_facilitates_and_competition_suppresses_target() -> None:
    positive = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    negative = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    masks = SignedMasks(positive, negative, torch.tensor([0, 1]))
    layer = SignedLateral(
        masks,
        cooperative_gain=1.0,
        competitive_gain=1.0,
        initial_magnitude=0.5,
    )
    output = layer(torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(output, torch.tensor([[3.5, 2.0]]), atol=1e-6)


def test_masked_weights_keep_their_intended_sign_and_gradients_are_finite() -> None:
    masks = structured_signed_masks(12, groups=3, seed=2)
    layer = SignedLateral(
        masks,
        cooperative_gain=0.25,
        competitive_gain=0.25,
        residual_scale=0.5,
    )
    features = torch.randn(7, 12, requires_grad=True)
    loss = layer(features).square().mean()
    loss.backward()
    assert torch.all(layer.cooperative_weights >= 0)
    assert torch.all(layer.competitive_weights >= 0)
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(layer.cooperative_raw.grad).all()
    assert torch.isfinite(layer.competitive_raw.grad).all()


def test_only_existing_edges_own_trainable_parameters() -> None:
    masks = structured_signed_masks(
        16,
        groups=4,
        cooperative_degree=2,
        competitive_degree=3,
    )
    layer = SignedLateral(masks)
    assert layer.cooperative_raw.numel() == int(masks.cooperative.sum())
    assert layer.competitive_raw.numel() == int(masks.competitive.sum())
