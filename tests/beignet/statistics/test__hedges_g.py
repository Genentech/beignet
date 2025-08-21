import hypothesis.strategies
import torch
from hypothesis import given, settings

import beignet
import beignet.statistics


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    n1=hypothesis.strategies.integers(min_value=5, max_value=20),
    n2=hypothesis.strategies.integers(min_value=5, max_value=20),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_hedges_g(batch_size, n1, n2, dtype):
    """Test Hedges' g effect size calculation."""
    # Generate test data
    group1 = torch.randn(batch_size, n1, dtype=dtype)
    group2 = torch.randn(batch_size, n2, dtype=dtype)

    # Test basic functionality
    result = beignet.statistics.hedges_g(group1, group2)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype

    # Test with identical groups (should be close to zero)
    identical_group = torch.randn(batch_size, n1, dtype=dtype)
    result_identical = beignet.statistics.hedges_g(identical_group, identical_group)
    assert torch.allclose(
        result_identical, torch.zeros_like(result_identical), atol=1e-6
    )

    # Test symmetry property: Hedges' g(A, B) = -Hedges' g(B, A)
    result_forward = beignet.statistics.hedges_g(group1, group2)
    result_backward = beignet.statistics.hedges_g(group2, group1)
    assert torch.allclose(result_forward, -result_backward, atol=1e-6)

    # Test with out parameter
    out = torch.empty(batch_size, dtype=dtype)
    result_out = beignet.statistics.hedges_g(group1, group2, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test relationship with Cohen's d
    cohens_d_result = beignet.statistics.cohens_d(group1, group2, pooled=True)
    hedges_g_result = beignet.statistics.hedges_g(group1, group2)

    # Hedges' g should be smaller in magnitude than Cohen's d (bias correction)
    assert torch.all(torch.abs(hedges_g_result) <= torch.abs(cohens_d_result))

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)
    result_grad = beignet.statistics.hedges_g(group1_grad, group2_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test torch.compile compatibility
    compiled_hedges_g = torch.compile(beignet.statistics.hedges_g, fullgraph=True)
    result_compiled = compiled_hedges_g(group1, group2)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test bias correction factor
    # For large sample sizes, Hedges' g should approach Cohen's d
    large_n1, large_n2 = 100, 100
    large_group1 = torch.randn(1, large_n1, dtype=dtype)
    large_group2 = torch.randn(1, large_n2, dtype=dtype)

    large_cohens_d = beignet.statistics.cohens_d(
        large_group1, large_group2, pooled=True
    )
    large_hedges_g = beignet.statistics.hedges_g(large_group1, large_group2)

    # Should be very close for large samples
    assert torch.allclose(large_cohens_d, large_hedges_g, atol=0.01)

    # For small sample sizes, bias correction should be more pronounced
    small_n1, small_n2 = 5, 5
    small_group1 = torch.randn(1, small_n1, dtype=dtype)
    small_group2 = torch.randn(1, small_n2, dtype=dtype) + 1.0  # Add effect

    small_cohens_d = beignet.statistics.cohens_d(
        small_group1, small_group2, pooled=True
    )
    small_hedges_g = beignet.statistics.hedges_g(small_group1, small_group2)

    # Hedges' g should be smaller in magnitude due to bias correction
    assert torch.abs(small_hedges_g) < torch.abs(small_cohens_d)

    # Test mathematical properties
    # For known distributions with effect size
    torch.manual_seed(42)
    normal1 = torch.normal(0.0, 1.0, size=(1, 50), dtype=dtype)
    normal2 = torch.normal(
        1.0, 1.0, size=(1, 50), dtype=dtype
    )  # Effect size should be ~1.0

    result_known = beignet.statistics.hedges_g(normal1, normal2)
    # Should be approximately -1.0 with bias correction (negative because normal2 > normal1)
    assert (
        torch.abs(result_known + 1.0) < 0.3
    )  # Allow variance due to sampling and bias correction
