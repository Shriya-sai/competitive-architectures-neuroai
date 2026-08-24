import torch
from torch.utils.data import TensorDataset

from competitive_architectures.cifar import _loader, _subset


def test_subset_and_loader_are_seed_reproducible() -> None:
    dataset = TensorDataset(torch.arange(20), torch.arange(20))
    first_subset = _subset(dataset, count=10, seed=4)
    second_subset = _subset(dataset, count=10, seed=4)
    assert first_subset.indices == second_subset.indices
    first_batches = [batch[0] for batch in _loader(first_subset, 4, 9, True)]
    second_batches = [batch[0] for batch in _loader(second_subset, 4, 9, True)]
    assert all(
        torch.equal(first, second)
        for first, second in zip(first_batches, second_batches, strict=True)
    )
