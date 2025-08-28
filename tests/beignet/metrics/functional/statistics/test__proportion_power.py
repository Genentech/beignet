import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    p0=st.floats(min_value=0.1, max_value=0.8),
    p1=st.floats(min_value=0.1, max_value=0.8),
    sample_size=st.integers(min_value=10, max_value=100),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_power(batch_size, p0, p1, sample_size, alpha, dtype):
    """Test proportion_power functional wrapper."""
    p0_tensor = torch.full((batch_size,), p0, dtype=dtype)
    p1_tensor = torch.full((batch_size,), p1, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    result_functional = beignet.metrics.functional.statistics.proportion_power(
        p0_tensor,
        p1_tensor,
        sample_size_tensor,
        alpha=alpha,
    )
    result_direct = beignet.statistics.proportion_power(
        p0_tensor,
        p1_tensor,
        sample_size_tensor,
        alpha=alpha,
    )

    # Results should be identical - handle edge cases where both might be NaN/inf
    if torch.all(torch.isfinite(result_functional)) and torch.all(
        torch.isfinite(result_direct),
    ):
        assert torch.allclose(result_functional, result_direct, atol=1e-6)
    else:
        # Both should have the same finite/non-finite pattern
        assert torch.equal(
            torch.isfinite(result_functional),
            torch.isfinite(result_direct),
        )

    assert isinstance(result_functional, torch.Tensor)
    assert result_functional.shape == (batch_size,)
    assert result_functional.dtype == dtype

    # Only check power bounds if results are finite
    if torch.all(torch.isfinite(result_functional)):
        assert torch.all(result_functional >= 0.0)
        assert torch.all(result_functional <= 1.0)
