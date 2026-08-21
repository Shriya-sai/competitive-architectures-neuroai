from competitive_architectures.synthetic import (
    run_correlated_mismatch_suite,
    run_synthetic_learning_experiment,
    run_synthetic_stress_suite,
)


def test_correct_topology_recovers_known_signed_mapping() -> None:
    result = run_synthetic_learning_experiment(samples=512, steps=200)
    assert result.structured_parameter_count == result.rewired_parameter_count
    assert result.final_structured_loss < result.initial_structured_loss * 1e-3
    assert result.final_structured_loss < result.final_rewired_loss * 1e-3


def test_stress_suite_includes_true_null_and_signed_signal() -> None:
    results = run_synthetic_stress_suite(
        seeds=(2, 7),
        teacher_magnitudes=(0.0, 0.2),
        target_noise_levels=(0.0,),
        sample_sizes=(256,),
        steps=150,
    )
    null, signed = results
    assert abs(null.mean_loss_advantage) < 1e-5
    assert signed.structured_wins == signed.seeds
    assert signed.mean_loss_advantage > 0.05


def test_correlated_mismatch_suite_has_identity_and_mismatch_conditions() -> None:
    results = run_correlated_mismatch_suite(
        seeds=(2, 7),
        input_correlations=(0.0, 0.9),
        rewiring_levels=(0.0, 10.0),
        samples=256,
        steps=150,
    )
    identity_results = [result for result in results if result.swaps_per_edge == 0]
    mismatch_results = [result for result in results if result.swaps_per_edge > 0]
    assert all(result.mean_edge_overlap == 1 for result in identity_results)
    assert all(abs(result.mean_loss_advantage) < 1e-10 for result in identity_results)
    assert all(result.structured_wins == result.seeds for result in mismatch_results)
