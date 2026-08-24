import torch

from competitive_architectures.lateral import SignedLateral
from competitive_architectures.models import paired_models, trainable_parameter_count


def test_core_models_share_output_shape_and_paired_backbone() -> None:
    models = paired_models(seed=8)
    images = torch.randn(4, 3, 32, 32)
    outputs = {name: model(images) for name, model in models.items()}
    assert all(output.shape == (4, 10) for output in outputs.values())
    standard_state = models["standard"].backbone.state_dict()
    for mode in ("random_signed", "structured_signed"):
        candidate_state = models[mode].backbone.state_dict()
        assert all(
            torch.equal(standard_state[name], candidate_state[name])
            for name in standard_state
        )


def test_random_and_structured_signed_models_are_capacity_matched() -> None:
    models = paired_models(seed=5)
    random_model = models["random_signed"]
    structured_model = models["structured_signed"]
    assert trainable_parameter_count(random_model) == trainable_parameter_count(
        structured_model
    )
    assert isinstance(random_model.interaction, SignedLateral)
    assert isinstance(structured_model.interaction, SignedLateral)
    assert random_model.interaction.cooperative_raw.numel() == 256
    assert random_model.interaction.competitive_raw.numel() == 256


def test_every_core_model_completes_a_finite_training_step() -> None:
    models = paired_models(seed=13)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    for model in models.values():
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
