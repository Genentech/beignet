import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from statsmodels.stats.effect_size import cohen_d_ind_samples

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    n1=st.integers(min_value=5, max_value=20),
    n2=st.integers(min_value=5, max_value=20),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_cohens_d(batch_size, n1, n2, dtype):
    """Test Cohen's d effect size calculation."""
    # Generate test data
    group1 = torch.randn(batch_size, n1, dtype=dtype)
    group2 = torch.randn(batch_size, n2, dtype=dtype)

    # Test basic functionality
    result = beignet.cohens_d(group1, group2, pooled=True)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype

    # Test non-pooled version
    result_nonpooled = beignet.cohens_d(group1, group2, pooled=False)
    assert result_nonpooled.shape == (batch_size,)
    assert result_nonpooled.dtype == dtype

    # Test with identical groups (should be close to zero)
    identical_group = torch.randn(batch_size, n1, dtype=dtype)
    result_identical = beignet.cohens_d(identical_group, identical_group, pooled=True)
    assert torch.allclose(
        result_identical, torch.zeros_like(result_identical), atol=1e-6
    )

    # Test symmetry property: Cohen's d(A, B) = -Cohen's d(B, A)
    result_forward = beignet.cohens_d(group1, group2, pooled=True)
    result_backward = beignet.cohens_d(group2, group1, pooled=True)
    assert torch.allclose(result_forward, -result_backward, atol=1e-6)

    # Test with out parameter
    out = torch.empty(batch_size, dtype=dtype)
    result_out = beignet.cohens_d(group1, group2, pooled=True, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)
    result_grad = beignet.cohens_d(group1_grad, group2_grad, pooled=True)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test torch.compile compatibility
    compiled_cohens_d = torch.compile(beignet.cohens_d, fullgraph=True)
    result_compiled = compiled_cohens_d(group1, group2, pooled=True)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test with different effect sizes
    # Large effect: mean difference = 2 * std
    large_effect_group1 = torch.zeros(batch_size, n1, dtype=dtype)
    large_effect_group2 = torch.ones(batch_size, n2, dtype=dtype) * 2.0

    result_large = beignet.cohens_d(
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

    result_known = beignet.cohens_d(normal1, normal2, pooled=True)
    # Should be approximately -1.0 (negative because normal2 > normal1)
    assert torch.abs(result_known + 1.0) < 0.5  # Allow some variance due to sampling


def test_cohens_d_against_statsmodels():
    """Test Cohen's d against statsmodels implementation."""
    if not HAS_STATSMODELS:
        return  # Skip if statsmodels not available

    torch.manual_seed(42)
    np.random.seed(42)

    # Generate test data
    n1, n2 = 30, 25
    group1_np = np.random.randn(n1)
    group2_np = np.random.randn(n2) + 0.5  # Add some effect

    # Convert to torch tensors
    group1_torch = torch.from_numpy(group1_np).float().unsqueeze(0)
    group2_torch = torch.from_numpy(group2_np).float().unsqueeze(0)

    # Compute using beignet
    beignet_result = beignet.cohens_d(group1_torch, group2_torch, pooled=True).item()

    # Compute using statsmodels
    statsmodels_result = cohen_d_ind_samples(
        group1_np, group2_np, pooled=True, bias_correction=False
    )

    # They should be very close
    assert abs(beignet_result - statsmodels_result) < 1e-6, (
        f"Beignet: {beignet_result}, Statsmodels: {statsmodels_result}"
    )

    # Test with different sample sizes
    group1_large = np.random.randn(100)
    group2_large = np.random.randn(80) + 0.3

    group1_large_torch = torch.from_numpy(group1_large).float().unsqueeze(0)
    group2_large_torch = torch.from_numpy(group2_large).float().unsqueeze(0)

    beignet_large = beignet.cohens_d(
        group1_large_torch, group2_large_torch, pooled=True
    ).item()
    statsmodels_large = cohen_d_ind_samples(
        group1_large, group2_large, pooled=True, bias_correction=False
    )

    assert abs(beignet_large - statsmodels_large) < 1e-6, (
        f"Large samples - Beignet: {beignet_large}, Statsmodels: {statsmodels_large}"
    )

    # Test non-pooled version (using unequal variance)
    beignet_unpooled = beignet.cohens_d(group1_torch, group2_torch, pooled=False).item()
    statsmodels_unpooled = cohen_d_ind_samples(
        group1_np, group2_np, pooled=False, bias_correction=False
    )

    assert abs(beignet_unpooled - statsmodels_unpooled) < 1e-6, (
        f"Unpooled - Beignet: {beignet_unpooled}, Statsmodels: {statsmodels_unpooled}"
    )
