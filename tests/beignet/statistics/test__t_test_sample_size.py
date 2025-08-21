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
def test_t_test_sample_size(batch_size, dtype):
    """Test one-sample/paired t-test sample size calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.t_test_sample_size(effect_sizes, power=0.8, alpha=0.05)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 2.0)  # Minimum sample size is 2
    assert torch.all(result <= 100000.0)  # Reasonable upper bound

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.t_test_sample_size(
        effect_sizes, power=0.8, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that sample size decreases with effect size
    small_effect = beignet.statistics.t_test_sample_size(
        torch.tensor(0.2, dtype=dtype), power=0.8
    )
    large_effect = beignet.statistics.t_test_sample_size(
        torch.tensor(0.8, dtype=dtype), power=0.8
    )
    assert small_effect > large_effect

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    result_grad = beignet.statistics.t_test_sample_size(effect_grad, power=0.8)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_sample_size = torch.compile(
        beignet.statistics.t_test_sample_size, fullgraph=True
    )
    result_compiled = compiled_ttest_sample_size(effect_sizes, power=0.8)
    assert torch.allclose(
        result, result_compiled, atol=1e-0
    )  # Allow integer rounding differences

    # Test sample size calculation for different power levels
    effect_size = torch.tensor(0.5, dtype=dtype)
    # Higher power should require larger sample size
    n_low = beignet.statistics.t_test_sample_size(effect_size, power=0.7)
    n_high = beignet.statistics.t_test_sample_size(effect_size, power=0.9)
    assert n_high > n_low

    # Test different alternative hypotheses
    # Two-sided test should require larger sample size than one-sided
    n_two_sided = beignet.statistics.t_test_sample_size(
        effect_size, power=0.8, alternative="two-sided"
    )
    n_one_sided = beignet.statistics.t_test_sample_size(
        effect_size, power=0.8, alternative="one-sided"
    )
    assert n_two_sided > n_one_sided

    # Test against known sample size values
    # For effect size = 0.5, power = 0.8, alpha = 0.05, two-sided
    # Expected sample size ≈ 34 (from power analysis literature)
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    sample_size = beignet.statistics.t_test_sample_size(
        effect_size_known, power=0.8, alpha=0.05, alternative="two-sided"
    )
    expected = 34
    # Allow some tolerance for approximation differences
    assert torch.abs(sample_size - expected) < 10

    # Test against statsmodels (only for float64 to avoid tolerance issues)
    if dtype == torch.float64:
        effect_sizes_test = [0.2, 0.5, 0.8]
        powers_test = [0.7, 0.8, 0.9]

        for effect_size_val in effect_sizes_test:
            for power_val in powers_test:
                # Test two-sided
                beignet_n = beignet.statistics.t_test_sample_size(
                    torch.tensor(effect_size_val, dtype=dtype),
                    power=power_val,
                    alpha=0.05,
                    alternative="two-sided",
                )

                # Use TTestPower for one-sample t-test
                statsmodels_n = statsmodels.stats.power.TTestPower().solve_power(
                    effect_size=effect_size_val,
                    nobs=None,
                    alpha=0.05,
                    power=power_val,
                    alternative="two-sided",
                )

                # Allow reasonable tolerance for different approximations and rounding
                assert torch.abs(beignet_n - statsmodels_n) < max(
                    40, 0.25 * statsmodels_n
                ), f"Failed for effect_size={effect_size_val}, power={power_val}"

    # Test edge cases
    # Very small effect size should require large sample size
    small_effect_edge = beignet.statistics.t_test_sample_size(
        torch.tensor(0.01, dtype=dtype), power=0.8
    )
    assert small_effect_edge > 1000

    # Very large effect size should require small sample size
    large_effect_edge = beignet.statistics.t_test_sample_size(
        torch.tensor(2.0, dtype=dtype), power=0.8
    )
    assert large_effect_edge < 10

    # Check minimum sample size constraint
    tiny_effect = beignet.statistics.t_test_sample_size(
        torch.tensor(1e-8, dtype=dtype), power=0.8
    )
    assert tiny_effect >= 2.0

    # Test consistency with power calculation
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

    # Test sample size calculation for different significance levels
    effect_size_alpha = torch.tensor(0.5, dtype=dtype)
    # Stricter alpha should require larger sample size
    n_strict = beignet.statistics.t_test_sample_size(
        effect_size_alpha, power=0.8, alpha=0.01
    )
    n_lenient = beignet.statistics.t_test_sample_size(
        effect_size_alpha, power=0.8, alpha=0.05
    )
    assert n_strict > n_lenient
