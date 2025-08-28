import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    groups=st.integers(min_value=3, max_value=8),
    sample_size=st.integers(min_value=10, max_value=50),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_analysis_of_variance_power(
    batch_size,
    effect_size,
    groups,
    sample_size,
    alpha,
    dtype,
):
    """Test analysis_of_variance_power functional wrapper."""
    # Create test inputs
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    groups_tensor = torch.full((batch_size,), groups, dtype=torch.int64)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    # Test functional wrapper
    result_functional = (
        beignet.metrics.functional.statistics.analysis_of_variance_power(
            effect_size_tensor,
            sample_size_tensor,
            groups_tensor,
            alpha=alpha,
        )
    )

    # Test direct call to beignet.statistics
    result_direct = beignet.statistics.analysis_of_variance_power(
        effect_size_tensor,
        sample_size_tensor,
        groups_tensor,
        alpha=alpha,
    )

    # Results should be identical
    assert torch.allclose(result_functional, result_direct, atol=1e-6)

    # Verify output properties
    assert isinstance(result_functional, torch.Tensor)
    assert result_functional.shape == (batch_size,)
    assert result_functional.dtype == dtype
    assert torch.all(result_functional >= 0.0)
    assert torch.all(result_functional <= 1.0)

    # Test that changing alpha can affect results (but skip if power saturated at 0 or 1)
    if alpha * 2 <= 0.1:
        result_different_alpha = (
            beignet.metrics.functional.statistics.analysis_of_variance_power(
                effect_size_tensor,
                sample_size_tensor,
                groups_tensor,
                alpha=alpha * 2,
            )
        )
        # Only test difference if power is in meaningful range (not saturated)
        power_in_range = torch.all(
            (result_functional > 0.01) & (result_functional < 0.99),
        )
        if power_in_range:
            assert not torch.allclose(
                result_functional,
                result_different_alpha,
                atol=1e-6,
            )

    # Test gradient computation
    effect_grad = effect_size_tensor.clone().requires_grad_(True)
    sample_grad = sample_size_tensor.clone().float().requires_grad_(True)
    groups_grad = groups_tensor.clone().float().requires_grad_(True)

    result_grad = beignet.metrics.functional.statistics.analysis_of_variance_power(
        effect_grad,
        sample_grad,
        groups_grad,
        alpha=alpha,
    )

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None
    assert groups_grad.grad is not None

    # Test edge cases - verify function doesn't crash with extreme values
    small_effect = torch.full((batch_size,), 0.01, dtype=dtype)
    power_small = beignet.metrics.functional.statistics.analysis_of_variance_power(
        small_effect,
        sample_size_tensor,
        groups_tensor,
        alpha=alpha,
    )
    assert torch.all(torch.isfinite(power_small))  # Should produce finite results

    large_effect = torch.full((batch_size,), 2.0, dtype=dtype)
    power_large = beignet.metrics.functional.statistics.analysis_of_variance_power(
        large_effect,
        sample_size_tensor,
        groups_tensor,
        alpha=alpha,
    )
    assert torch.all(torch.isfinite(power_large))  # Should produce finite results
