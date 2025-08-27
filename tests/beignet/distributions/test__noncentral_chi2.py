import pytest
import torch

import beignet.distributions


def test_noncentral_chi2_basic():
    """Test basic NonCentralChi2 functionality."""
    dtype = torch.float64

    # Test basic functionality
    df = torch.tensor(5.0, dtype=dtype)
    nc = torch.tensor(2.0, dtype=dtype)

    dist = beignet.distributions.NonCentralChi2(df, nc)

    # Test that it's a Distribution
    assert isinstance(dist, torch.distributions.Distribution)

    # Test icdf with various probability values
    probabilities = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], dtype=dtype)
    quantiles = dist.icdf(probabilities)

    # Test output shape and bounds
    assert quantiles.shape == (7,)
    assert torch.all(quantiles >= 0.0)  # Chi-squared is non-negative
    assert torch.all(torch.isfinite(quantiles))

    # Test monotonicity: icdf should be monotonically increasing
    for j in range(6):
        assert quantiles[j] <= quantiles[j + 1], (
            f"Non-monotonic at indices {j}, {j + 1}"
        )

    # Test that mean and variance are reasonable
    mean = dist.mean
    variance = dist.variance

    expected_mean = df + nc  # Theoretical mean
    expected_variance = 2 * (df + 2 * nc)  # Theoretical variance

    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(variance, expected_variance)


def test_noncentral_chi2_central_case():
    """Test NonCentralChi2 with zero non-centrality (central case)."""
    dtype = torch.float64

    # Central chi-squared (nc = 0)
    df = torch.tensor(3.0, dtype=dtype)
    nc = torch.tensor(0.0, dtype=dtype)

    dist = beignet.distributions.NonCentralChi2(df, nc)

    # Test that it behaves like central chi-squared
    probabilities = torch.tensor([0.05, 0.5, 0.95], dtype=dtype)
    quantiles = dist.icdf(probabilities)

    # For central chi-squared with df=3:
    # 5th percentile ≈ 0.35, median ≈ 2.37, 95th percentile ≈ 7.81
    expected_approx = torch.tensor([0.35, 2.37, 7.81], dtype=dtype)

    # Allow reasonable tolerance for our approximation
    assert torch.allclose(quantiles, expected_approx, rtol=0.3)

    # Test monotonicity
    assert quantiles[0] < quantiles[1] < quantiles[2]


def test_noncentral_chi2_parameter_validation():
    """Test parameter validation."""
    dtype = torch.float64

    # Test valid parameters
    dist = beignet.distributions.NonCentralChi2(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(0.5, dtype=dtype),
        validate_args=True,
    )
    assert dist.df.item() == 1.0
    assert dist.nc.item() == 0.5

    # Test that negative df raises error
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        beignet.distributions.NonCentralChi2(
            torch.tensor(-1.0, dtype=dtype),
            torch.tensor(1.0, dtype=dtype),
            validate_args=True,
        )

    # Test that negative nc raises error
    with pytest.raises(
        ValueError, match="Non-centrality parameter must be non-negative"
    ):
        beignet.distributions.NonCentralChi2(
            torch.tensor(1.0, dtype=dtype),
            torch.tensor(-1.0, dtype=dtype),
            validate_args=True,
        )


def test_noncentral_chi2_edge_cases():
    """Test edge cases and extreme parameters."""
    dtype = torch.float64

    # Small degrees of freedom
    small_dist = beignet.distributions.NonCentralChi2(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
    )

    small_quantiles = small_dist.icdf(torch.tensor([0.1, 0.5, 0.9], dtype=dtype))
    assert torch.all(torch.isfinite(small_quantiles))
    assert torch.all(small_quantiles >= 0.0)

    # Large non-centrality
    large_nc_dist = beignet.distributions.NonCentralChi2(
        torch.tensor(5.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
    )

    large_quantiles = large_nc_dist.icdf(torch.tensor([0.1, 0.5, 0.9], dtype=dtype))
    assert torch.all(torch.isfinite(large_quantiles))
    assert torch.all(large_quantiles >= 0.0)

    # The quantiles should be much larger due to high non-centrality
    assert torch.all(large_quantiles > small_quantiles)


def test_noncentral_chi2_batched():
    """Test NonCentralChi2 with batched parameters."""
    dtype = torch.float64

    # Batched parameters
    df_batch = torch.tensor([1.0, 3.0, 5.0], dtype=dtype)
    nc_batch = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)

    dist = beignet.distributions.NonCentralChi2(df_batch, nc_batch)

    # Test batch shape
    assert dist.batch_shape == (3,)

    # Single probability, multiple distributions
    prob = torch.tensor(0.5, dtype=dtype)
    medians = dist.icdf(prob)

    assert medians.shape == (3,)
    assert torch.all(torch.isfinite(medians))
    assert torch.all(medians >= 0.0)

    # Multiple probabilities, multiple distributions
    probs = torch.tensor([0.25, 0.5, 0.75], dtype=dtype)
    quantiles = dist.icdf(probs)

    assert quantiles.shape == (3,)
    assert torch.all(torch.isfinite(quantiles))
    assert torch.all(quantiles >= 0.0)


def test_noncentral_chi2_torch_compile():
    """Test torch.compile compatibility."""
    dtype = torch.float64

    df = torch.tensor(4.0, dtype=dtype)
    nc = torch.tensor(1.5, dtype=dtype)
    dist = beignet.distributions.NonCentralChi2(df, nc)

    # Test torch.compile compatibility
    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    test_prob = torch.tensor(0.3, dtype=dtype)
    original_result = dist.icdf(test_prob)
    compiled_result = compiled_icdf(test_prob)

    assert torch.allclose(original_result, compiled_result, rtol=1e-10)


def test_noncentral_chi2_gradients():
    """Test gradient computation."""
    dtype = torch.float64

    # Test gradients with respect to parameters
    df_grad = torch.tensor(3.0, dtype=dtype, requires_grad=True)
    nc_grad = torch.tensor(1.0, dtype=dtype, requires_grad=True)

    dist = beignet.distributions.NonCentralChi2(df_grad, nc_grad)

    prob = torch.tensor(0.6, dtype=dtype)
    quantile = dist.icdf(prob)

    # Test that gradients can be computed
    quantile.backward()

    assert df_grad.grad is not None
    assert nc_grad.grad is not None
    assert torch.isfinite(df_grad.grad)
    assert torch.isfinite(nc_grad.grad)


def test_noncentral_chi2_properties():
    """Test distribution properties."""
    dtype = torch.float64

    # Test various parameter combinations
    test_cases = [
        (1.0, 0.0),  # Central case, df=1
        (3.0, 0.0),  # Central case, df=3
        (1.0, 2.0),  # Non-central, small df
        (5.0, 1.0),  # Non-central, moderate parameters
        (10.0, 5.0),  # Non-central, larger parameters
    ]

    for df_val, nc_val in test_cases:
        df = torch.tensor(df_val, dtype=dtype)
        nc = torch.tensor(nc_val, dtype=dtype)

        dist = beignet.distributions.NonCentralChi2(df, nc)

        # Test mean and variance properties
        mean = dist.mean
        variance = dist.variance

        expected_mean = df + nc
        expected_var = 2 * (df + 2 * nc)

        assert torch.allclose(mean, expected_mean)
        assert torch.allclose(variance, expected_var)

        # Test some quantiles
        quantiles = dist.icdf(torch.tensor([0.1, 0.5, 0.9], dtype=dtype))

        # Basic sanity checks
        assert torch.all(torch.isfinite(quantiles))
        assert torch.all(quantiles >= 0.0)
        assert quantiles[0] < quantiles[1] < quantiles[2]  # Monotonic


if __name__ == "__main__":
    test_noncentral_chi2_basic()
    test_noncentral_chi2_central_case()
    test_noncentral_chi2_parameter_validation()
    test_noncentral_chi2_edge_cases()
    test_noncentral_chi2_batched()
    test_noncentral_chi2_torch_compile()
    test_noncentral_chi2_gradients()
    test_noncentral_chi2_properties()
    print("All NonCentralChi2 tests passed!")
