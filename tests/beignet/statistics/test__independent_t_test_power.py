import hypothesis.strategies
import statsmodels.stats.power
import torch
from hypothesis import given, settings

import beignet
import beignet.statistics


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_independent_t_test_power(batch_size, dtype):
    """Test independent samples t-test power calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    nobs1_values = (
        torch.tensor([20, 30, 50], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    ratio_values = (
        torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.independent_t_test_power(
        effect_sizes, nobs1_values, ratio_values, alpha=0.05
    )
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.independent_t_test_power(
        effect_sizes, nobs1_values, ratio_values, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test with default ratio
    result_default = beignet.statistics.independent_t_test_power(
        effect_sizes, nobs1_values, alpha=0.05
    )
    assert result_default.shape == effect_sizes.shape

    # Test that power increases with effect size
    small_effect = beignet.statistics.independent_t_test_power(
        torch.tensor(0.2, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    large_effect = beignet.statistics.independent_t_test_power(
        torch.tensor(0.8, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.statistics.independent_t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(20.0, dtype=dtype)
    )
    large_n = beignet.statistics.independent_t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(80.0, dtype=dtype)
    )
    assert large_n > small_n

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    nobs1_grad = nobs1_values.clone().requires_grad_(True)
    ratio_grad = ratio_values.clone().requires_grad_(True)
    result_grad = beignet.statistics.independent_t_test_power(
        effect_grad, nobs1_grad, ratio_grad
    )

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert nobs1_grad.grad is not None
    assert ratio_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_ind_power = torch.compile(
        beignet.statistics.independent_t_test_power, fullgraph=True
    )
    result_compiled = compiled_ttest_ind_power(effect_sizes, nobs1_values, ratio_values)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test different alternative hypotheses
    effect_size = torch.tensor(0.5, dtype=dtype)
    nobs1 = torch.tensor(30.0, dtype=dtype)
    # Two-sided test should have lower power than one-sided for same parameters
    power_two_sided = beignet.statistics.independent_t_test_power(
        effect_size, nobs1, alternative="two-sided"
    )
    power_larger = beignet.statistics.independent_t_test_power(
        effect_size, nobs1, alternative="larger"
    )
    power_smaller = beignet.statistics.independent_t_test_power(
        effect_size, nobs1, alternative="smaller"
    )
    assert power_two_sided <= power_larger
    # For positive effect size, "smaller" should have very low power
    assert power_smaller <= power_larger

    # Test against known power values
    # For effect size = 0.5, n1 = n2 = 30, alpha = 0.05, two-sided
    # Expected power ≈ 0.47 (from power analysis literature)
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    nobs1_known = torch.tensor(30.0, dtype=dtype)
    power_known = beignet.statistics.independent_t_test_power(
        effect_size_known, nobs1_known, alpha=0.05, alternative="two-sided"
    )
    expected = 0.47
    # Allow some tolerance for approximation differences
    assert torch.abs(power_known - expected) < 0.1

    # Test against statsmodels (only for float64 to avoid tolerance issues)
    if dtype == torch.float64:
        effect_sizes_test = [0.2, 0.5, 0.8]
        nobs1_values_test = [20, 30, 50]
        ratios_test = [1.0, 1.5]

        for effect_size_val in effect_sizes_test:
            for nobs1_val in nobs1_values_test:
                for ratio_val in ratios_test:
                    # Test two-sided
                    beignet_power = beignet.statistics.independent_t_test_power(
                        torch.tensor(effect_size_val, dtype=dtype),
                        torch.tensor(float(nobs1_val), dtype=dtype),
                        torch.tensor(ratio_val, dtype=dtype),
                        alpha=0.05,
                        alternative="two-sided",
                    )

                    # Use tt_ind_solve_power for independent samples t-test
                    statsmodels_power = statsmodels.stats.power.tt_ind_solve_power(
                        effect_size=effect_size_val,
                        nobs1=nobs1_val,
                        alpha=0.05,
                        power=None,
                        ratio=ratio_val,
                        alternative="two-sided",
                    )

                    # Allow reasonable tolerance for different approximations
                    assert torch.abs(beignet_power - statsmodels_power) < 0.15, (
                        f"Failed for effect_size={effect_size_val}, nobs1={nobs1_val}, ratio={ratio_val}"
                    )

    # Test edge cases
    # Zero effect size should give power ≈ alpha
    zero_effect = beignet.statistics.independent_t_test_power(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        alpha=0.05,
    )
    assert torch.abs(zero_effect - 0.05) < 0.03

    # Very large effect size should give power ≈ 1
    large_effect_edge = beignet.statistics.independent_t_test_power(
        torch.tensor(2.0, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    assert large_effect_edge > 0.9

    # Very small sample size (minimum is 2)
    small_n_edge = beignet.statistics.independent_t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),  # Will be clamped to 2
    )
    assert 0.0 <= small_n_edge <= 1.0

    # Test effects of different sample size ratios
    effect_size_ratio = torch.tensor(0.5, dtype=dtype)
    nobs1_ratio = torch.tensor(30.0, dtype=dtype)
    # Balanced design (ratio=1) should generally have higher power than unbalanced
    power_balanced = beignet.statistics.independent_t_test_power(
        effect_size_ratio, nobs1_ratio, torch.tensor(1.0, dtype=dtype)
    )
    power_unbalanced = beignet.statistics.independent_t_test_power(
        effect_size_ratio, nobs1_ratio, torch.tensor(0.5, dtype=dtype)
    )
    # For same total sample size, balanced is more powerful
    assert power_balanced > power_unbalanced

    # Test consistency with sample size calculation
    effect_size_consistency = torch.tensor(0.5, dtype=dtype)
    ratio_consistency = torch.tensor(1.0, dtype=dtype)
    target_power = 0.8
    # Calculate required sample size
    nobs1_consistency = beignet.statistics.independent_t_test_sample_size(
        effect_size_consistency, ratio_consistency, power=target_power
    )
    # Calculate power with that sample size
    achieved_power = beignet.statistics.independent_t_test_power(
        effect_size_consistency, nobs1_consistency, ratio_consistency
    )
    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.1
