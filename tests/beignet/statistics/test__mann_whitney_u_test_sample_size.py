import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_mann_whitney_u_test_sample_size(batch_size, dtype):
    """Test Mann-Whitney U test sample size calculation."""

    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality (equal group sizes)
    result = beignet.statistics.mann_whitney_u_test_sample_size(effect_sizes, power=0.8)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 1.0)  # Sample size should be at least 1

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.mann_whitney_u_test_sample_size(
        effect_sizes,
        power=0.8,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that sample size decreases with larger effect size
    small_effect = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.1, dtype=dtype),
        power=0.8,
    )
    large_effect = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(1.0, dtype=dtype),
        power=0.8,
    )
    assert small_effect >= large_effect  # Allow equal due to discretization

    # Test that sample size increases with higher power requirement
    low_power = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.5, dtype=dtype),
        power=0.7,
    )
    high_power = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.5, dtype=dtype),
        power=0.9,
    )
    assert high_power >= low_power  # Allow equal due to discretization

    # Test with ratio parameter (unequal group sizes)
    equal_groups = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.5, dtype=dtype),
        ratio=1.0,
        power=0.8,
    )
    unequal_groups = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.5, dtype=dtype),
        ratio=2.0,
        power=0.8,
    )
    # Unequal groups typically require larger total sample size
    assert unequal_groups >= equal_groups * 0.8  # Allow some flexibility

    # Test different alpha values
    alpha_05 = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.5, dtype=dtype),
        power=0.8,
        alpha=0.05,
    )
    alpha_01 = beignet.statistics.mann_whitney_u_test_sample_size(
        torch.tensor(0.5, dtype=dtype),
        power=0.8,
        alpha=0.01,
    )
    assert (
        alpha_01 >= alpha_05
    )  # More stringent alpha requires larger sample size (allow equal due to constraints)

    # Test gradient computation
    effect_grad = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    result_grad = beignet.statistics.mann_whitney_u_test_sample_size(
        effect_grad,
        power=0.8,
    )

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert torch.all(torch.isfinite(effect_grad.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        beignet.statistics.mann_whitney_u_test_sample_size,
        fullgraph=True,
    )
    result_compiled = compiled_func(effect_sizes[:1], power=0.8)
    result_normal = beignet.statistics.mann_whitney_u_test_sample_size(
        effect_sizes[:1],
        power=0.8,
    )
    assert torch.allclose(result_compiled, result_normal, atol=1e-6)
