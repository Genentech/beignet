import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    sample_size_group1=st.integers(min_value=5, max_value=30),
    sample_size_group2=st.integers(min_value=5, max_value=30),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_hedges_g(batch_size, sample_size_group1, sample_size_group2, dtype):
    """Test hedges_g functional wrapper."""
    group1 = torch.randn(batch_size, sample_size_group1, dtype=dtype)
    group2 = torch.randn(batch_size, sample_size_group2, dtype=dtype)

    result_functional = beignet.metrics.functional.statistics.hedges_g(group1, group2)
    result_direct = beignet.statistics.hedges_g(group1, group2)

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert result_functional.shape == (batch_size,)
    assert result_functional.dtype == dtype

    # Test symmetry
    result_backward = beignet.metrics.functional.statistics.hedges_g(group2, group1)
    assert torch.allclose(result_functional, -result_backward, atol=1e-6)
