import hypothesis
import hypothesis.strategies
import statsmodels.stats.power
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_t_test_power(batch_size, dtype):
    """Test one-sample/paired t-test power calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.t_test_power(effect_sizes, sample_sizes, alpha=0.05)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.t_test_power(
        effect_sizes, sample_sizes, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.statistics.t_test_power(
        torch.tensor(0.2, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    large_effect = beignet.statistics.t_test_power(
        torch.tensor(0.8, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.statistics.t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(20.0, dtype=dtype)
    )
    large_n = beignet.statistics.t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(100.0, dtype=dtype)
    )
    assert large_n > small_n

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    sample_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.statistics.t_test_power(effect_grad, sample_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_power = torch.compile(
        beignet.statistics.t_test_power, fullgraph=True
    )
    result_compiled = compiled_ttest_power(effect_sizes, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test different alternative hypotheses
    effect_size = torch.tensor(0.5, dtype=dtype)
    sample_size = torch.tensor(30.0, dtype=dtype)
    # Two-sided test should have lower power than one-sided for same parameters
    power_two_sided = beignet.statistics.t_test_power(
        effect_size, sample_size, alternative="two-sided"
    )
    power_one_sided = beignet.statistics.t_test_power(
        effect_size, sample_size, alternative="one-sided"
    )
    assert power_two_sided < power_one_sided

    # Test against known power values
    # For effect size = 0.5, n = 30, alpha = 0.05, two-sided
    # Expected power ≈ 0.659 (from power analysis literature)
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    sample_size_known = torch.tensor(30.0, dtype=dtype)
    power_known = beignet.statistics.t_test_power(
        effect_size_known, sample_size_known, alpha=0.05, alternative="two-sided"
    )
    expected = 0.659
    # Allow some tolerance for approximation differences
    assert torch.abs(power_known - expected) < 0.05

    # Test against statsmodels (only for float64 to avoid tolerance issues)
    if dtype == torch.float64:
        effect_sizes_test = [0.2, 0.5, 0.8]
        sample_sizes_test = [20, 30, 50]

        for effect_size_val in effect_sizes_test:
            for sample_size_val in sample_sizes_test:
                # Test two-sided
                beignet_power = beignet.statistics.t_test_power(
                    torch.tensor(effect_size_val, dtype=dtype),
                    torch.tensor(float(sample_size_val), dtype=dtype),
                    alpha=0.05,
                    alternative="two-sided",
                )

                # Use TTestPower for one-sample t-test
                statsmodels_power = statsmodels.stats.power.TTestPower().solve_power(
                    effect_size=effect_size_val,
                    nobs=sample_size_val,
                    alpha=0.05,
                    power=None,
                    alternative="two-sided",
                )

                # Allow reasonable tolerance for different approximations
                assert torch.abs(beignet_power - statsmodels_power) < 0.1, (
                    f"Failed for effect_size={effect_size_val}, sample_size={sample_size_val}"
                )

    # Test edge cases
    # Zero effect size should give power ≈ alpha
    zero_effect = beignet.statistics.t_test_power(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        alpha=0.05,
    )
    assert torch.abs(zero_effect - 0.05) < 0.025

    # Very large effect size should give power ≈ 1
    large_effect_edge = beignet.statistics.t_test_power(
        torch.tensor(3.0, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    assert large_effect_edge > 0.99

    # Very small sample size (minimum is 2)
    small_n_edge = beignet.statistics.t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),  # Will be clamped to 2
    )
    assert 0.0 <= small_n_edge <= 1.0

    # Test consistency with sample size calculation
    effect_size_consistency = torch.tensor(0.5, dtype=dtype)
    target_power = 0.8
    # Calculate required sample size
    sample_size_consistency = beignet.statistics.t_test_sample_size(
        effect_size_consistency, power=target_power
    )
    # Calculate power with that sample size
    achieved_power = beignet.statistics.t_test_power(
        effect_size_consistency, sample_size_consistency
    )
    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.05
