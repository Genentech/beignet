import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_cohens_f_squared(batch_size, dtype):
    """Test Cohen's f² effect size calculation."""
    # Generate test parameters - create group means for 3 groups
    group_means = (
        torch.tensor(
            [[10.0, 12.0, 14.0], [5.0, 7.0, 9.0], [20.0, 22.0, 24.0]], dtype=dtype
        )
        .repeat(batch_size, 1, 1)
        .view(-1, 3)
    )

    pooled_stds = (
        torch.tensor([2.0, 1.5, 3.0], dtype=dtype).repeat(batch_size).flatten()
    )

    # Test basic functionality
    result = beignet.cohens_f_squared(group_means, pooled_stds)
    assert result.shape == pooled_stds.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)

    # Test with out parameter
    out = torch.empty_like(pooled_stds)
    result_out = beignet.cohens_f_squared(group_means, pooled_stds, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test relationship to Cohen's f
    cohens_f_values = beignet.cohens_f(group_means, pooled_stds)
    expected_f_squared = cohens_f_values**2
    assert torch.allclose(result, expected_f_squared, atol=1e-6)

    # Test that Cohen's f² increases with larger spread in group means
    close_means = torch.tensor([10.0, 10.1, 10.2], dtype=dtype)
    spread_means = torch.tensor([10.0, 15.0, 20.0], dtype=dtype)
    pooled_std = torch.tensor(2.0, dtype=dtype)

    f2_close = beignet.cohens_f_squared(close_means, pooled_std)
    f2_spread = beignet.cohens_f_squared(spread_means, pooled_std)

    assert f2_spread > f2_close

    # Test gradient computation
    group_means_grad = group_means.clone().requires_grad_(True)
    pooled_stds_grad = pooled_stds.clone().requires_grad_(True)
    result_grad = beignet.cohens_f_squared(group_means_grad, pooled_stds_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group_means_grad.grad is not None
    assert pooled_stds_grad.grad is not None

    # Test torch.compile compatibility
    compiled_cohens_f_squared = torch.compile(beignet.cohens_f_squared, fullgraph=True)
    result_compiled = compiled_cohens_f_squared(group_means, pooled_stds)
    assert torch.allclose(result, result_compiled, atol=1e-6)


def test_cohens_f_squared_known_values():
    """Test Cohen's f² against known values."""
    # Example from Cohen (1988)
    # Three groups with means [10, 12, 14] and pooled std = 2
    group_means = torch.tensor([10.0, 12.0, 14.0], dtype=torch.float32)
    pooled_std = torch.tensor(2.0, dtype=torch.float32)

    f2 = beignet.cohens_f_squared(group_means, pooled_std)

    # Manual calculation: f = 0.8165, so f² ≈ 0.6667
    expected = (1.633 / 2.0) ** 2
    assert torch.abs(f2 - expected) < 0.01

    # Test interpretation guidelines
    # Small effect: f² = 0.01
    small_means = torch.tensor(
        [10.0, 10.14, 10.28], dtype=torch.float32
    )  # Adjusted for small f²
    pooled_std_small = torch.tensor(2.0, dtype=torch.float32)
    f2_small = beignet.cohens_f_squared(small_means, pooled_std_small)
    assert f2_small < 0.04  # Should be small effect

    # Medium effect: f² = 0.0625
    medium_means = torch.tensor([10.0, 10.5, 11.0], dtype=torch.float32)
    pooled_std_medium = torch.tensor(2.0, dtype=torch.float32)
    f2_medium = beignet.cohens_f_squared(medium_means, pooled_std_medium)
    assert 0.03 < f2_medium < 0.15  # Should be around medium effect


def test_cohens_f_squared_eta_squared_relationship():
    """Test relationship between Cohen's f² and eta-squared."""
    # η² = f² / (1 + f²) and f² = η² / (1 - η²)
    group_means = torch.tensor([10.0, 12.0, 14.0], dtype=torch.float64)
    pooled_std = torch.tensor(2.0, dtype=torch.float64)

    f2 = beignet.cohens_f_squared(group_means, pooled_std)

    # Calculate eta-squared from f²
    eta_squared = f2 / (1 + f2)

    # Convert back to f² to verify relationship
    f2_from_eta = eta_squared / (1 - eta_squared)

    assert torch.abs(f2 - f2_from_eta) < 1e-10


def test_cohens_f_squared_edge_cases():
    """Test edge cases for Cohen's f² calculation."""
    # Test with identical group means (should give f² = 0)
    identical_means = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    pooled_std = torch.tensor(2.0, dtype=torch.float64)
    f2_identical = beignet.cohens_f_squared(identical_means, pooled_std)
    assert torch.abs(f2_identical) < 1e-10

    # Test with very small pooled std
    group_means = torch.tensor([10.0, 12.0, 14.0], dtype=torch.float64)
    tiny_std = torch.tensor(1e-12, dtype=torch.float64)
    f2_tiny_std = beignet.cohens_f_squared(group_means, tiny_std)
    assert torch.isfinite(f2_tiny_std)
    assert f2_tiny_std > 1000000  # Should be very large

    # Test with single group (edge case)
    single_group = torch.tensor([10.0], dtype=torch.float64)
    f2_single = beignet.cohens_f_squared(single_group, pooled_std)
    assert torch.abs(f2_single) < 1e-10  # Standard deviation of single value is 0


def test_cohens_f_squared_manual_calculation():
    """Test Cohen's f² with manual calculation."""
    # Test case with known outcome
    group_means = torch.tensor([8.0, 10.0, 12.0], dtype=torch.float64)
    pooled_std = torch.tensor(2.0, dtype=torch.float64)

    # Manual calculation
    # f = 0.8164965809277261 (from Cohen's f test)
    # f² = 0.8164965809277261² ≈ 0.6666666666666666

    result = beignet.cohens_f_squared(group_means, pooled_std)
    expected = 0.6666666666666666

    assert torch.abs(result - expected) < 1e-10
