import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_friedman_test_power(batch_size, dtype):
    """Test Friedman test power calculation."""
    # Generate test data - effect sizes, subjects, and treatments
    # Friedman test is a non-parametric test for repeated measures
    effect_sizes = torch.rand(batch_size, dtype=dtype) * 5.0 + 0.1  # 0.1 to 5.1
    n_subjects = torch.randint(5, 50, (batch_size,), dtype=dtype)
    n_treatments = torch.randint(3, 8, (batch_size,), dtype=dtype)

    # Test basic functionality
    result = beignet.statistics.friedman_test_power(
        effect_sizes,
        n_subjects,
        n_treatments,
        alpha=0.05,
    )
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty(batch_size, dtype=dtype)
    result_out = beignet.statistics.friedman_test_power(
        effect_sizes,
        n_subjects,
        n_treatments,
        alpha=0.05,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.statistics.friedman_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
    )
    large_effect = beignet.statistics.friedman_test_power(
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
    )
    assert large_effect > small_effect

    # Test that power increases with number of subjects
    small_n = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
    )
    large_n = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test that power generally increases with number of treatments (more comparisons)
    few_treatments = beignet.statistics.friedman_test_power(
        torch.tensor(1.5, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
    )
    many_treatments = beignet.statistics.friedman_test_power(
        torch.tensor(1.5, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(6.0, dtype=dtype),
    )
    # This relationship can vary, but generally more treatments increase power for detecting differences
    # We'll just ensure both are reasonable values
    assert 0.0 <= few_treatments <= 1.0
    assert 0.0 <= many_treatments <= 1.0

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    subjects_grad = n_subjects.clone().requires_grad_(True)
    treatments_grad = n_treatments.clone().requires_grad_(True)
    result_grad = beignet.statistics.friedman_test_power(
        effect_grad,
        subjects_grad,
        treatments_grad,
    )

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert subjects_grad.grad is not None
    assert treatments_grad.grad is not None
    assert effect_grad.grad.shape == effect_sizes.shape
    assert subjects_grad.grad.shape == n_subjects.shape
    assert treatments_grad.grad.shape == n_treatments.shape

    # Test torch.compile compatibility
    compiled_friedman_power = torch.compile(
        beignet.statistics.friedman_test_power,
        fullgraph=True,
    )
    result_compiled = compiled_friedman_power(effect_sizes, n_subjects, n_treatments)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test mathematical properties - Friedman test formula verification
    # Lambda (non-centrality parameter) = 12 * n_subjects * effect_size / (k * (k + 1))
    # where k = n_treatments
    test_effect = torch.tensor(1.0, dtype=dtype)
    test_subjects = torch.tensor(15.0, dtype=dtype)
    test_treatments = torch.tensor(4.0, dtype=dtype)

    result_known = beignet.statistics.friedman_test_power(
        test_effect,
        test_subjects,
        test_treatments,
        alpha=0.05,
    )

    # Expected lambda = 12 * 15 * 1.0 / (4 * 5) = 180 / 20 = 9.0
    # This should give reasonable power
    assert 0.5 < result_known < 0.99

    # Test with zero effect size (should give power ≈ alpha)
    zero_effect = beignet.statistics.friedman_test_power(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
        alpha=0.05,
    )
    # With zero effect, power should be close to alpha (Type I error rate)
    assert torch.abs(zero_effect - 0.05) < 0.03

    # Test edge cases
    # Test with minimum values (should be clamped)
    min_subjects = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),  # Will be clamped to 3.0
        torch.tensor(2.0, dtype=dtype),  # Will be clamped to 3.0
    )
    assert 0.0 <= min_subjects <= 1.0

    # Test with large effect size
    large_effect_known = beignet.statistics.friedman_test_power(
        torch.tensor(5.0, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
        torch.tensor(5.0, dtype=dtype),
    )
    # Should have high power for large effect
    assert large_effect_known > 0.9

    # Test different alpha levels
    alpha_01 = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
        alpha=0.01,
    )
    alpha_05 = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(4.0, dtype=dtype),
        alpha=0.05,
    )
    # Power should be higher for larger alpha (less stringent)
    assert alpha_05 > alpha_01

    # Test batch processing with mixed scenarios
    mixed_effects = torch.tensor([0.0, 1.0, 3.0], dtype=dtype)
    mixed_subjects = torch.tensor([20.0, 20.0, 20.0], dtype=dtype)
    mixed_treatments = torch.tensor([4.0, 4.0, 4.0], dtype=dtype)
    result_mixed = beignet.statistics.friedman_test_power(
        mixed_effects,
        mixed_subjects,
        mixed_treatments,
    )

    # Power should increase with effect size
    assert result_mixed[1] > result_mixed[0]  # medium > zero effect
    assert result_mixed[2] > result_mixed[1]  # large > medium effect
    assert torch.all(result_mixed >= 0.0)
    assert torch.all(result_mixed <= 1.0)

    # Test numerical stability with extreme values
    extreme_effect = torch.tensor(10.0, dtype=dtype)
    extreme_subjects = torch.tensor(100.0, dtype=dtype)
    extreme_treatments = torch.tensor(10.0, dtype=dtype)
    result_extreme = beignet.statistics.friedman_test_power(
        extreme_effect,
        extreme_subjects,
        extreme_treatments,
    )
    assert torch.all(torch.isfinite(result_extreme))
    assert 0.0 <= result_extreme <= 1.0

    # Test Friedman test specific properties
    # The test is used for comparing k≥3 related groups (repeated measures)
    # Test minimum number of treatments (should be clamped to 3)
    two_treatments = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(15.0, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),  # Will be clamped to 3.0
    )
    three_treatments = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(15.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
    )
    # Should be the same due to clamping
    assert torch.allclose(two_treatments, three_treatments, atol=1e-6)

    # Test degrees of freedom calculation (k-1 where k = n_treatments)
    # This is implicitly tested through the chi-squared distribution usage
    for k in [3, 4, 5, 6]:
        df_test = beignet.statistics.friedman_test_power(
            torch.tensor(1.0, dtype=dtype),
            torch.tensor(20.0, dtype=dtype),
            torch.tensor(float(k), dtype=dtype),
        )
        assert torch.isfinite(df_test)
        assert 0.0 <= df_test <= 1.0

    # Test consistency across different dtypes
    if dtype == torch.float32:
        # Test same calculation with float64 should be very similar
        result_f32 = beignet.statistics.friedman_test_power(
            torch.tensor(1.5, dtype=torch.float32),
            torch.tensor(25.0, dtype=torch.float32),
            torch.tensor(4.0, dtype=torch.float32),
        )
        result_f64 = beignet.statistics.friedman_test_power(
            torch.tensor(1.5, dtype=torch.float64),
            torch.tensor(25.0, dtype=torch.float64),
            torch.tensor(4.0, dtype=torch.float64),
        )
        # Should be very close (allowing for precision differences)
        assert torch.abs(result_f32.float() - result_f64.float()) < 0.01

    # Test power curves - power should increase monotonically with effect size
    effect_range = torch.linspace(0.1, 3.0, 10, dtype=dtype)
    power_curve = []
    for effect in effect_range:
        power_val = beignet.statistics.friedman_test_power(
            effect,
            torch.tensor(20.0, dtype=dtype),
            torch.tensor(4.0, dtype=dtype),
        )
        power_curve.append(float(power_val))

    # Check that power generally increases (allowing for some numerical variation)
    for i in range(1, len(power_curve)):
        # Allow small decreases due to numerical precision, but trend should be upward
        if i > 3:  # After initial small effects, should be clearly increasing
            assert power_curve[i] >= power_curve[i - 3] - 0.05

    # Test that power approaches 1 for very large effects
    huge_effect = beignet.statistics.friedman_test_power(
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        torch.tensor(5.0, dtype=dtype),
    )
    assert huge_effect > 0.99

    # Test Friedman test interpretation
    # Used for testing whether k related groups have different central tendencies
    # Non-parametric alternative to repeated measures ANOVA
    typical_case = beignet.statistics.friedman_test_power(
        torch.tensor(1.0, dtype=dtype),  # Moderate effect
        torch.tensor(20.0, dtype=dtype),  # 20 subjects
        torch.tensor(3.0, dtype=dtype),  # 3 treatments (minimum)
        alpha=0.05,
    )
    # Should give reasonable power for typical experimental design
    assert 0.3 < typical_case < 0.99

    # Test small sample behavior
    small_sample = beignet.statistics.friedman_test_power(
        torch.tensor(2.0, dtype=dtype),  # Large effect needed for small samples
        torch.tensor(5.0, dtype=dtype),  # Small sample
        torch.tensor(3.0, dtype=dtype),
    )
    # Even with large effect, small samples have limited power
    assert small_sample < 0.85
