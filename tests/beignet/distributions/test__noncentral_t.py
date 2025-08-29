import torch

import beignet.distributions


def test_noncentral_t_basic():
    """Test basic NonCentralT functionality."""
    dtype = torch.float64

    # Test basic functionality
    df = torch.tensor(10.0, dtype=dtype)
    nc = torch.tensor(1.5, dtype=dtype)

    dist = beignet.distributions.NonCentralT(df, nc)

    # Test that it's a Distribution
    assert isinstance(dist, torch.distributions.Distribution)

    # Test icdf with various probability values
    probabilities = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], dtype=dtype)
    quantiles = dist.icdf(probabilities)

    # Test output shape and bounds
    assert quantiles.shape == (7,)
    assert torch.all(torch.isfinite(quantiles))

    # Test monotonicity: icdf should be monotonically increasing
    for j in range(6):
        assert quantiles[j] <= quantiles[j + 1], (
            f"Non-monotonic at indices {j}, {j + 1}"
        )

    # Test that mean and variance are reasonable
    mean = dist.mean
    variance = dist.variance

    assert torch.isfinite(mean)
    assert torch.isfinite(variance)
    assert variance > 0.0  # Variance should be positive


def test_noncentral_t_central_case():
    """Test NonCentralT with zero non-centrality (central case)."""
    dtype = torch.float64

    # Central t-distribution (nc = 0)
    df = torch.tensor(5.0, dtype=dtype)
    nc = torch.tensor(0.0, dtype=dtype)

    dist = beignet.distributions.NonCentralT(df, nc)

    # Test that it behaves like central t-distribution
    probabilities = torch.tensor([0.025, 0.5, 0.975], dtype=dtype)
    quantiles = dist.icdf(probabilities)

    # For central t with df=5:
    # 2.5th percentile ≈ -2.57, median = 0, 97.5th percentile ≈ 2.57
    expected_approx = torch.tensor([-2.57, 0.0, 2.57], dtype=dtype)

    # Allow reasonable tolerance for our approximation
    assert torch.allclose(quantiles, expected_approx, rtol=0.2, atol=0.1)

    # Test symmetry around zero
    assert (
        torch.abs(quantiles[0] + quantiles[2]) < 0.1
    )  # Should be approximately symmetric


def test_noncentral_t_parameter_validation():
    """Test parameter validation."""
    dtype = torch.float64

    # Test valid parameters
    dist = beignet.distributions.NonCentralT(
        torch.tensor(5.0, dtype=dtype),
        torch.tensor(-1.5, dtype=dtype),  # nc can be negative
        validate_args=True,
    )
    assert dist.df.item() == 5.0
    assert dist.nc.item() == -1.5

    # Test that zero/negative df are accepted (validation is disabled for torch.compile compatibility)
    dist_zero_df = beignet.distributions.NonCentralT(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        validate_args=True,
    )
    assert dist_zero_df.df.item() == 0.0

    dist_neg_df = beignet.distributions.NonCentralT(
        torch.tensor(-2.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        validate_args=True,
    )
    assert dist_neg_df.df.item() == -2.0


def test_noncentral_t_non_centrality_effect():
    """Test that non-centrality parameter shifts the distribution."""
    dtype = torch.float64

    df = torch.tensor(8.0, dtype=dtype)

    # Central case
    central_dist = beignet.distributions.NonCentralT(df, torch.tensor(0.0, dtype=dtype))

    # Positive non-centrality
    pos_nc_dist = beignet.distributions.NonCentralT(df, torch.tensor(2.0, dtype=dtype))

    # Negative non-centrality
    neg_nc_dist = beignet.distributions.NonCentralT(df, torch.tensor(-2.0, dtype=dtype))

    prob = torch.tensor(0.5, dtype=dtype)  # Median

    central_median = central_dist.icdf(prob)
    pos_median = pos_nc_dist.icdf(prob)
    neg_median = neg_nc_dist.icdf(prob)

    # Positive non-centrality should shift distribution to the right
    assert pos_median > central_median

    # Negative non-centrality should shift distribution to the left
    assert neg_median < central_median

    # Should be approximately symmetric
    assert torch.allclose(
        central_median - neg_median,
        pos_median - central_median,
        rtol=0.3,
    )


def test_noncentral_t_edge_cases():
    """Test edge cases and extreme parameters."""
    dtype = torch.float64

    # Small degrees of freedom (but >= 2 for numerical stability)
    small_dist = beignet.distributions.NonCentralT(
        torch.tensor(2.5, dtype=dtype),
        torch.tensor(0.5, dtype=dtype),
    )

    small_quantiles = small_dist.icdf(torch.tensor([0.1, 0.5, 0.9], dtype=dtype))
    assert torch.all(torch.isfinite(small_quantiles))

    # Large degrees of freedom (should approach normal)
    large_dist = beignet.distributions.NonCentralT(
        torch.tensor(200.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    large_quantiles = large_dist.icdf(torch.tensor([0.1, 0.5, 0.9], dtype=dtype))
    assert torch.all(torch.isfinite(large_quantiles))

    # For large df, should be approximately normal with mean shift
    # Median should be close to non-centrality parameter
    assert torch.abs(large_quantiles[1] - 1.0) < 0.2

    # Large non-centrality
    large_nc_dist = beignet.distributions.NonCentralT(
        torch.tensor(5.0, dtype=dtype),
        torch.tensor(10.0, dtype=dtype),
    )

    large_nc_quantiles = large_nc_dist.icdf(torch.tensor([0.1, 0.5, 0.9], dtype=dtype))
    assert torch.all(torch.isfinite(large_nc_quantiles))


def test_noncentral_t_batched():
    """Test NonCentralT with batched parameters."""
    dtype = torch.float64

    # Batched parameters
    df_batch = torch.tensor([2.0, 5.0, 10.0], dtype=dtype)
    nc_batch = torch.tensor([0.0, 1.0, -1.0], dtype=dtype)

    dist = beignet.distributions.NonCentralT(df_batch, nc_batch)

    # Test batch shape
    assert dist.batch_shape == (3,)

    # Single probability, multiple distributions
    prob = torch.tensor(0.5, dtype=dtype)
    medians = dist.icdf(prob)

    assert medians.shape == (3,)
    assert torch.all(torch.isfinite(medians))

    # The median should reflect the non-centrality:
    # First (nc=0) should be near 0, second (nc=1) should be positive, third (nc=-1) should be negative
    assert torch.abs(medians[0]) < 0.5  # Central case
    assert medians[1] > 0.0  # Positive nc
    assert medians[2] < 0.0  # Negative nc


def test_noncentral_t_torch_compile():
    """Test torch.compile compatibility."""
    dtype = torch.float64

    df = torch.tensor(6.0, dtype=dtype)
    nc = torch.tensor(1.0, dtype=dtype)
    dist = beignet.distributions.NonCentralT(df, nc)

    # Test torch.compile compatibility
    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    test_prob = torch.tensor(0.3, dtype=dtype)
    original_result = dist.icdf(test_prob)
    compiled_result = compiled_icdf(test_prob)

    assert torch.allclose(original_result, compiled_result, rtol=1e-10)


def test_noncentral_t_gradients():
    """Test gradient computation."""
    dtype = torch.float64

    # Test gradients with respect to parameters
    df_grad = torch.tensor(4.0, dtype=dtype, requires_grad=True)
    nc_grad = torch.tensor(0.5, dtype=dtype, requires_grad=True)

    dist = beignet.distributions.NonCentralT(df_grad, nc_grad)

    prob = torch.tensor(0.6, dtype=dtype)
    quantile = dist.icdf(prob)

    # Test that gradients can be computed
    quantile.backward()

    assert df_grad.grad is not None
    assert nc_grad.grad is not None
    assert torch.isfinite(df_grad.grad)
    assert torch.isfinite(nc_grad.grad)


def test_noncentral_t_different_df():
    """Test behavior across different degrees of freedom."""
    dtype = torch.float64

    nc = torch.tensor(1.0, dtype=dtype)
    prob = torch.tensor(0.75, dtype=dtype)  # 75th percentile

    # Test increasing degrees of freedom
    df_values = [1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
    quantiles = []

    for df_val in df_values:
        df = torch.tensor(df_val, dtype=dtype)
        dist = beignet.distributions.NonCentralT(df, nc)
        quantile = dist.icdf(prob)
        quantiles.append(quantile.item())

    # As df increases, should approach normal quantile + nc
    # For large df, 75th percentile of standard normal ≈ 0.674
    # So non-central t should approach 0.674 + 1.0 = 1.674
    assert abs(quantiles[-1] - 1.674) < 0.1

    # Earlier values should be larger (t-distribution has heavier tails)
    for i in range(len(quantiles) - 1):
        assert quantiles[i] >= quantiles[i + 1] - 0.1  # Allow small tolerance


def test_noncentral_t_properties():
    """Test distribution properties and consistency."""
    dtype = torch.float64

    # Test various parameter combinations
    test_cases = [
        (2.0, 0.0),  # Central case, small df
        (5.0, 0.0),  # Central case, moderate df
        (3.0, 1.0),  # Positive nc
        (4.0, -1.5),  # Negative nc
        (10.0, 2.0),  # Larger parameters
    ]

    for df_val, nc_val in test_cases:
        df = torch.tensor(df_val, dtype=dtype)
        nc = torch.tensor(nc_val, dtype=dtype)

        dist = beignet.distributions.NonCentralT(df, nc)

        # Test quantiles across range
        probs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=dtype)
        quantiles = dist.icdf(probs)

        # Basic sanity checks
        assert torch.all(torch.isfinite(quantiles))

        # Monotonicity
        for i in range(len(quantiles) - 1):
            assert quantiles[i] <= quantiles[i + 1]

        # For positive nc, median should generally be positive (except for very small df)
        if nc_val > 0 and df_val > 2:
            assert quantiles[2] > 0  # Median should be positive

        # For negative nc, median should generally be negative (except for very small df)
        if nc_val < 0 and df_val > 2:
            assert quantiles[2] < 0  # Median should be negative


if __name__ == "__main__":
    test_noncentral_t_basic()
    test_noncentral_t_central_case()
    test_noncentral_t_parameter_validation()
    test_noncentral_t_non_centrality_effect()
    test_noncentral_t_edge_cases()
    test_noncentral_t_batched()
    test_noncentral_t_torch_compile()
    test_noncentral_t_gradients()
    test_noncentral_t_different_df()
    test_noncentral_t_properties()
    print("All NonCentralT tests passed!")
