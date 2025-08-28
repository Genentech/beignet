import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_kruskal_wallis_test_power(batch_size, dtype):
    """Test Kruskal-Wallis test power calculation."""

    # Generate test parameters - need at least 3 groups for Kruskal-Wallis
    effect_sizes = torch.tensor([0.1, 0.3, 0.5], dtype=dtype)[:batch_size]
    # Sample sizes for 3 groups each
    sample_sizes = torch.tensor(
        [[20, 30, 40], [25, 35, 45], [30, 40, 50]],
        dtype=dtype,
    )[:batch_size]

    # Test basic functionality
    result = beignet.statistics.kruskal_wallis_test_power(
        effect_sizes,
        sample_sizes,
    )
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.kruskal_wallis_test_power(
        effect_sizes,
        sample_sizes,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.statistics.kruskal_wallis_test_power(
        torch.tensor(0.1, dtype=dtype),
        torch.tensor([30.0, 40.0, 50.0], dtype=dtype),
    )
    large_effect = beignet.statistics.kruskal_wallis_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor([30.0, 40.0, 50.0], dtype=dtype),
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.statistics.kruskal_wallis_test_power(
        torch.tensor(0.3, dtype=dtype),
        torch.tensor([10.0, 15.0, 20.0], dtype=dtype),
    )
    large_n = beignet.statistics.kruskal_wallis_test_power(
        torch.tensor(0.3, dtype=dtype),
        torch.tensor([40.0, 50.0, 60.0], dtype=dtype),
    )
    assert large_n > small_n

    # Test different alpha values
    power_05 = beignet.statistics.kruskal_wallis_test_power(
        torch.tensor(0.3, dtype=dtype),
        torch.tensor([30.0, 40.0, 50.0], dtype=dtype),
        alpha=0.05,
    )
    power_01 = beignet.statistics.kruskal_wallis_test_power(
        torch.tensor(0.3, dtype=dtype),
        torch.tensor([30.0, 40.0, 50.0], dtype=dtype),
        alpha=0.01,
    )
    assert power_05 > power_01
