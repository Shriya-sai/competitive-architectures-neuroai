"""Competitive architectures and NeuroAI evaluation tools."""

from competitive_architectures.cifar import CifarSmokeResult, run_cifar10_smoke
from competitive_architectures.continual import ContinualResult, run_split_cifar10_pilot
from competitive_architectures.graphs import (
    SignedMasks,
    edge_overlap_fraction,
    rewire_signed_masks,
    signed_degrees,
    structured_signed_masks,
)
from competitive_architectures.lateral import (
    GatedSignedLateral,
    SignedBottleneck,
    SignedLateral,
)
from competitive_architectures.models import (
    TinyCifarCNN,
    paired_models,
    trainable_parameter_count,
)
from competitive_architectures.multiseed import run_frozen_multiseed
from competitive_architectures.synthetic import (
    CorrelatedMismatchResult,
    StressConditionResult,
    SyntheticLearningResult,
    run_correlated_mismatch_suite,
    run_synthetic_learning_experiment,
    run_synthetic_stress_suite,
)
from competitive_architectures.topology import TopologyAlignment, topology_alignment

__version__ = "0.1.0"

__all__ = [
    "CifarSmokeResult",
    "ContinualResult",
    "CorrelatedMismatchResult",
    "GatedSignedLateral",
    "SignedBottleneck",
    "SignedLateral",
    "SignedMasks",
    "StressConditionResult",
    "SyntheticLearningResult",
    "TinyCifarCNN",
    "TopologyAlignment",
    "edge_overlap_fraction",
    "paired_models",
    "rewire_signed_masks",
    "run_cifar10_smoke",
    "run_correlated_mismatch_suite",
    "run_frozen_multiseed",
    "run_split_cifar10_pilot",
    "run_synthetic_learning_experiment",
    "run_synthetic_stress_suite",
    "signed_degrees",
    "structured_signed_masks",
    "topology_alignment",
    "trainable_parameter_count",
]
