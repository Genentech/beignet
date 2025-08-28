import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    sample_size_group1=st.integers(min_value=5, max_value=30),
    sample_size_group2=st.integers(min_value=5, max_value=30),
    pooled=st.booleans(),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cohens_d(batch_size, sample_size_group1, sample_size_group2, pooled, dtype):
    """Test cohens_d functional wrapper."""
    group1 = torch.randn(batch_size, sample_size_group1, dtype=dtype)
    group2 = torch.randn(batch_size, sample_size_group2, dtype=dtype)

    result_functional = beignet.metrics.functional.statistics.cohens_d(
        group1,
        group2,
        pooled=pooled,
    )

    result_direct = beignet.statistics.cohens_d(
        group1,
        group2,
        pooled=pooled,
    )

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert result_functional.shape == (batch_size,)
    assert result_functional.dtype == dtype

    # Test symmetry: Cohen's d(A, B) = -Cohen's d(B, A) only when pooled=True
    # When pooled=False, symmetry doesn't hold due to different denominators
    if pooled:
        group_a = torch.randn(batch_size, sample_size_group1, dtype=dtype)
        group_b = (
            torch.randn(batch_size, sample_size_group2, dtype=dtype) + 1.0
        )  # Different mean

        result_ab = beignet.metrics.functional.statistics.cohens_d(
            group_a,
            group_b,
            pooled=pooled,
        )
        result_ba = beignet.metrics.functional.statistics.cohens_d(
            group_b,
            group_a,
            pooled=pooled,
        )

        # Only test symmetry if results are finite (avoid NaN issues)
        valid_mask = torch.isfinite(result_ab) & torch.isfinite(result_ba)
        if torch.any(valid_mask):
            assert torch.allclose(
                result_ab[valid_mask],
                -result_ba[valid_mask],
                atol=1e-5,
            )
