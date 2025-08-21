import hypothesis.strategies
import torch
from hypothesis import given, settings

import beignet
import beignet.statistics

# from statsmodels.stats.effect_size import cohen_d_ind_samples  # Function not available in current statsmodels version


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    n1=hypothesis.strategies.integers(min_value=5, max_value=20),
    n2=hypothesis.strategies.integers(min_value=5, max_value=20),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_cohens_d(batch_size, n1, n2, dtype):
    """Test Cohen's d effect size calculation."""
    # Generate test data
    group1 = torch.randn(batch_size, n1, dtype=dtype)
    group2 = torch.randn(batch_size, n2, dtype=dtype)

    # Test basic functionality
    result = beignet.statistics.cohens_d(group1, group2, pooled=True)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype

    # Test non-pooled version
    result_nonpooled = beignet.statistics.cohens_d(group1, group2, pooled=False)
    assert result_nonpooled.shape == (batch_size,)
    assert result_nonpooled.dtype == dtype

    # Test with identical groups (should be close to zero)
    identical_group = torch.randn(batch_size, n1, dtype=dtype)
    result_identical = beignet.statistics.cohens_d(
        identical_group, identical_group, pooled=True
    )
    assert torch.allclose(
        result_identical, torch.zeros_like(result_identical), atol=1e-6
    )

    # Test symmetry property: Cohen's d(A, B) = -Cohen's d(B, A)
    result_forward = beignet.statistics.cohens_d(group1, group2, pooled=True)
    result_backward = beignet.statistics.cohens_d(group2, group1, pooled=True)
    assert torch.allclose(result_forward, -result_backward, atol=1e-6)

    # Test with out parameter
    out = torch.empty(batch_size, dtype=dtype)
    result_out = beignet.statistics.cohens_d(group1, group2, pooled=True, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)
    result_grad = beignet.statistics.cohens_d(group1_grad, group2_grad, pooled=True)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test torch.compile compatibility
    compiled_cohens_d = torch.compile(beignet.statistics.cohens_d, fullgraph=True)
    result_compiled = compiled_cohens_d(group1, group2, pooled=True)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test with different effect sizes
    # Large effect: mean difference = 2 * std
    large_effect_group1 = torch.zeros(batch_size, n1, dtype=dtype)
    large_effect_group2 = torch.ones(batch_size, n2, dtype=dtype) * 2.0

    result_large = beignet.statistics.cohens_d(
        large_effect_group1, large_effect_group2, pooled=True
    )
    # Should be approximately -2.0 (negative because group2 > group1)
    assert torch.all(result_large < 0)

    # Test mathematical properties
    # For known distributions
    torch.manual_seed(42)
    normal1 = torch.normal(0.0, 1.0, size=(1, 100), dtype=dtype)
    normal2 = torch.normal(
        1.0, 1.0, size=(1, 100), dtype=dtype
    )  # Effect size should be ~1.0

    result_known = beignet.statistics.cohens_d(normal1, normal2, pooled=True)
    # Should be approximately -1.0 (negative because normal2 > normal1)
    assert torch.abs(result_known + 1.0) < 0.5  # Allow some variance due to sampling


# def test_cohens_d_against_statsmodels():
#     """Test Cohen's d against statsmodels implementation."""
#     # Disabled: cohen_d_ind_samples function not available in current statsmodels version
#     pass
