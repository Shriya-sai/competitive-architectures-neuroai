"""Competitive architectures and NeuroAI evaluation tools."""

from competitive_architectures.graphs import (
    SignedMasks,
    rewire_signed_masks,
    signed_degrees,
    structured_signed_masks,
)
from competitive_architectures.lateral import SignedLateral
from competitive_architectures.synthetic import (
    StressConditionResult,
    SyntheticLearningResult,
    run_synthetic_learning_experiment,
    run_synthetic_stress_suite,
)

__version__ = "0.1.0"

__all__ = [
    "SignedLateral",
    "SignedMasks",
    "StressConditionResult",
    "SyntheticLearningResult",
    "rewire_signed_masks",
    "run_synthetic_learning_experiment",
    "run_synthetic_stress_suite",
    "signed_degrees",
    "structured_signed_masks",
]
