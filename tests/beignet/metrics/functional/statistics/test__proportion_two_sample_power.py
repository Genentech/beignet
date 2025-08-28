import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    p1=st.floats(min_value=0.1, max_value=0.8),
    p2=st.floats(min_value=0.1, max_value=0.8),
    n1=st.integers(min_value=10, max_value=100),
    n2=st.integers(min_value=10, max_value=100),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_two_sample_power(batch_size, p1, p2, n1, n2, alpha, dtype):
    """Test proportion_two_sample_power functional wrapper."""
    p1_tensor = torch.full((batch_size,), p1, dtype=dtype)
    p2_tensor = torch.full((batch_size,), p2, dtype=dtype)
    n1_tensor = torch.full((batch_size,), n1, dtype=torch.int64)
    n2_tensor = torch.full((batch_size,), n2, dtype=torch.int64)

    result_functional = (
        beignet.metrics.functional.statistics.proportion_two_sample_power(
            p1_tensor,
            p2_tensor,
            n1_tensor,
            n2_tensor,
            alpha=alpha,
        )
    )
    result_direct = beignet.statistics.proportion_two_sample_power(
        p1_tensor,
        p2_tensor,
        n1_tensor,
        n2_tensor,
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
