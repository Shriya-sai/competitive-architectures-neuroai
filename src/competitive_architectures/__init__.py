"""Competitive architectures and NeuroAI evaluation tools."""

from competitive_architectures.graphs import (
    SignedMasks,
    edge_overlap_fraction,
    rewire_signed_masks,
    signed_degrees,
    structured_signed_masks,
)
from competitive_architectures.lateral import SignedLateral
from competitive_architectures.synthetic import (
    CorrelatedMismatchResult,
    StressConditionResult,
    SyntheticLearningResult,
    run_correlated_mismatch_suite,
    run_synthetic_learning_experiment,
    run_synthetic_stress_suite,
)

__version__ = "0.1.0"

__all__ = [
    "CorrelatedMismatchResult",
    "SignedLateral",
    "SignedMasks",
    "StressConditionResult",
    "SyntheticLearningResult",
    "edge_overlap_fraction",
    "rewire_signed_masks",
    "run_correlated_mismatch_suite",
    "run_synthetic_learning_experiment",
    "run_synthetic_stress_suite",
    "signed_degrees",
    "structured_signed_masks",
]
