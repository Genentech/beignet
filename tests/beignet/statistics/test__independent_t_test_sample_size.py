import hypothesis.strategies
import statsmodels.stats.power as smp
import torch
from hypothesis import given, settings

import beignet
import beignet.statistics


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_independent_t_test_sample_size(batch_size, dtype):
    """Test independent samples t-test sample size calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    ratio_values = (
        torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.independent_t_test_sample_size(
        effect_sizes, ratio_values, power=0.8, alpha=0.05
    )
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 2.0)  # Minimum sample size is 2
    assert torch.all(result <= 100000.0)  # Reasonable upper bound

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.independent_t_test_sample_size(
        effect_sizes, ratio_values, power=0.8, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test with default ratio
    result_default = beignet.statistics.independent_t_test_sample_size(
        effect_sizes, power=0.8, alpha=0.05
    )
    assert result_default.shape == effect_sizes.shape

    # Test that sample size decreases with effect size
    small_effect = beignet.statistics.independent_t_test_sample_size(
        torch.tensor(0.2, dtype=dtype), power=0.8
    )
    large_effect = beignet.statistics.independent_t_test_sample_size(
        torch.tensor(0.8, dtype=dtype), power=0.8
    )
    assert small_effect > large_effect

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    ratio_grad = ratio_values.clone().requires_grad_(True)
    result_grad = beignet.statistics.independent_t_test_sample_size(
        effect_grad, ratio_grad, power=0.8
    )

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert ratio_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_ind_sample_size = torch.compile(
        beignet.statistics.independent_t_test_sample_size, fullgraph=True
    )
    result_compiled = compiled_ttest_ind_sample_size(
        effect_sizes, ratio_values, power=0.8
    )
    assert torch.allclose(
        result, result_compiled, atol=1e-0
    )  # Allow integer rounding differences

    # Test sample size calculation for different power levels
    effect_size = torch.tensor(0.5, dtype=dtype)
    ratio = torch.tensor(1.0, dtype=dtype)

    # Higher power should require larger sample size
    n_low = beignet.statistics.independent_t_test_sample_size(
        effect_size, ratio, power=0.7
    )
    n_high = beignet.statistics.independent_t_test_sample_size(
        effect_size, ratio, power=0.9
    )
    assert n_high > n_low

    # Test different alternative hypotheses
    # Two-sided test should require larger sample size than one-sided
    n_two_sided = beignet.statistics.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alternative="two-sided"
    )
    n_larger = beignet.statistics.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alternative="larger"
    )
    assert n_two_sided > n_larger

    # Test against known sample size values
    # For effect size = 0.5, power = 0.8, alpha = 0.05, two-sided, ratio = 1
    # Expected sample size per group ≈ 64 (from power analysis literature)
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    ratio_known = torch.tensor(1.0, dtype=dtype)
    sample_size_known = beignet.statistics.independent_t_test_sample_size(
        effect_size_known, ratio_known, power=0.8, alpha=0.05, alternative="two-sided"
    )
    expected = 64
    # Allow some tolerance for approximation differences
    assert torch.abs(sample_size_known - expected) < 15

    # Test against statsmodels (only for float64 to avoid tolerance issues)
    if dtype == torch.float64:
        effect_sizes_test = [0.2, 0.5, 0.8]
        powers_test = [0.7, 0.8, 0.9]
        ratios_test = [1.0, 1.5]

        for effect_size_val in effect_sizes_test:
            for power_val in powers_test:
                for ratio_val in ratios_test:
                    # Test two-sided
                    beignet_n = beignet.statistics.independent_t_test_sample_size(
                        torch.tensor(effect_size_val, dtype=dtype),
                        torch.tensor(ratio_val, dtype=dtype),
                        power=power_val,
                        alpha=0.05,
                        alternative="two-sided",
                    )

                    # Use tt_ind_solve_power for independent samples t-test
                    statsmodels_n = smp.tt_ind_solve_power(
                        effect_size=effect_size_val,
                        nobs1=None,
                        alpha=0.05,
                        power=power_val,
                        ratio=ratio_val,
                        alternative="two-sided",
                    )

                    # Allow reasonable tolerance for different approximations and rounding
                    assert torch.abs(beignet_n - statsmodels_n) < max(
                        80, 0.3 * statsmodels_n
                    ), (
                        f"Failed for effect_size={effect_size_val}, power={power_val}, ratio={ratio_val}"
                    )

    # Test edge cases
    # Very small effect size should require large sample size
    small_effect_edge = beignet.statistics.independent_t_test_sample_size(
        torch.tensor(0.01, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        power=0.8,
    )
    assert small_effect_edge > 1000

    # Very large effect size should require small sample size
    large_effect_edge = beignet.statistics.independent_t_test_sample_size(
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        power=0.8,
    )
    assert large_effect_edge < 10

    # Check minimum sample size constraint
    tiny_effect = beignet.statistics.independent_t_test_sample_size(
        torch.tensor(1e-8, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        power=0.8,
    )
    assert tiny_effect >= 2.0

    # Test effects of different sample size ratios
    effect_size_ratio = torch.tensor(0.5, dtype=dtype)
    # Balanced design (ratio=1) should require smallest sample size
    n1_balanced = beignet.statistics.independent_t_test_sample_size(
        effect_size_ratio, torch.tensor(1.0, dtype=dtype), power=0.8
    )
    n1_unbalanced = beignet.statistics.independent_t_test_sample_size(
        effect_size_ratio, torch.tensor(0.5, dtype=dtype), power=0.8
    )
    # For same power, balanced design is more efficient
    assert n1_balanced < n1_unbalanced

    # Test consistency with power calculation
    effect_size_consistency = torch.tensor(0.5, dtype=dtype)
    ratio_consistency = torch.tensor(1.0, dtype=dtype)
    target_power = 0.8
    # Calculate required sample size
    nobs1 = beignet.statistics.independent_t_test_sample_size(
        effect_size_consistency, ratio_consistency, power=target_power
    )
    # Calculate power with that sample size
    achieved_power = beignet.statistics.independent_t_test_power(
        effect_size_consistency, nobs1, ratio_consistency
    )
    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.1

    # Test sample size calculation for different significance levels
    effect_size_alpha = torch.tensor(0.5, dtype=dtype)
    ratio_alpha = torch.tensor(1.0, dtype=dtype)
    # Stricter alpha should require larger sample size
    n_strict = beignet.statistics.independent_t_test_sample_size(
        effect_size_alpha, ratio_alpha, power=0.8, alpha=0.01
    )
    n_lenient = beignet.statistics.independent_t_test_sample_size(
        effect_size_alpha, ratio_alpha, power=0.8, alpha=0.05
    )
    assert n_strict > n_lenient
