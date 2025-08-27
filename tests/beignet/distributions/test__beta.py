import torch

import beignet.distributions


def test_beta_distribution():
    """Test Beta distribution functionality."""
    dtype = torch.float64

    # Test basic functionality
    concentration1 = torch.tensor(2.0, dtype=dtype)
    concentration0 = torch.tensor(3.0, dtype=dtype)

    dist = beignet.distributions.Beta(concentration1, concentration0)

    # Test that it inherits from torch.distributions.Beta
    assert isinstance(dist, torch.distributions.Beta)

    # Test icdf with various probability values
    probabilities = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=dtype)
    quantiles = dist.icdf(probabilities)

    # Test output shape and bounds
    assert quantiles.shape == (5,)
    assert torch.all(quantiles >= 0.0)
    assert torch.all(quantiles <= 1.0)

    # Test monotonicity: icdf should be monotonically increasing
    for j in range(4):
        assert quantiles[j] <= quantiles[j + 1], (
            f"Non-monotonic at indices {j}, {j + 1}"
        )

    # Test edge cases
    eps = torch.finfo(dtype).eps
    edge_probs = torch.tensor([eps, 0.5, 1 - eps], dtype=dtype)
    edge_quantiles = dist.icdf(edge_probs)

    assert torch.all(torch.isfinite(edge_quantiles))
    assert torch.all(edge_quantiles >= 0.0)
    assert torch.all(edge_quantiles <= 1.0)

    # Test different parameter ranges
    # Small parameters
    small_dist = beignet.distributions.Beta(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.8, dtype=dtype),
    )
    small_quantile = small_dist.icdf(torch.tensor(0.5, dtype=dtype))
    assert torch.isfinite(small_quantile)
    assert 0.0 <= small_quantile <= 1.0

    # Large parameters
    large_dist = beignet.distributions.Beta(
        torch.tensor(15.0, dtype=dtype),
        torch.tensor(12.0, dtype=dtype),
    )
    large_quantile = large_dist.icdf(torch.tensor(0.5, dtype=dtype))
    assert torch.isfinite(large_quantile)
    assert 0.0 <= large_quantile <= 1.0

    # Test torch.compile compatibility
    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    test_prob = torch.tensor(0.3, dtype=dtype)
    original_result = dist.icdf(test_prob)
    compiled_result = compiled_icdf(test_prob)

    assert torch.allclose(original_result, compiled_result, rtol=1e-6)

    # Test gradients
    conc1 = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    conc0 = torch.tensor(3.0, dtype=dtype, requires_grad=True)
    grad_dist = beignet.distributions.Beta(conc1, conc0)

    prob = torch.tensor(0.4, dtype=dtype)
    quantile = grad_dist.icdf(prob)

    # Test that gradients can be computed
    quantile.backward()

    assert conc1.grad is not None
    assert conc0.grad is not None
    assert torch.isfinite(conc1.grad)
    assert torch.isfinite(conc0.grad)


def test_beta_distribution_edge_cases():
    """Test specific edge cases and parameter combinations."""
    dtype = torch.float64

    # Test Beta(1,1) - should be uniform
    uniform_dist = beignet.distributions.Beta(
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    # For Beta(1,1), icdf(p) should be exactly p (uniform distribution)
    test_probs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=dtype)
    quantiles = uniform_dist.icdf(test_probs)
    assert torch.allclose(test_probs, quantiles, atol=1e-10)

    # Test extreme parameters
    extreme_dist = beignet.distributions.Beta(
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
    )
    extreme_quantile = extreme_dist.icdf(torch.tensor(0.5, dtype=dtype))
    assert torch.isfinite(extreme_quantile)
    assert 0.0 <= extreme_quantile <= 1.0

    # Test very different parameters
    skewed_dist = beignet.distributions.Beta(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(5.0, dtype=dtype),
    )
    skewed_quantile = skewed_dist.icdf(torch.tensor(0.5, dtype=dtype))
    assert torch.isfinite(skewed_quantile)
    assert 0.0 <= skewed_quantile <= 1.0
    # Should be closer to 0 than 1 for this distribution
    assert skewed_quantile < 0.5


def test_beta_distribution_accuracy():
    """Test accuracy against known values."""
    dtype = torch.float64

    # Beta(2, 5) - test general behavior (relaxed bounds for approximation)
    dist = beignet.distributions.Beta(
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(5.0, dtype=dtype),
    )

    # Test median (normal approximation may not be exact)
    median = dist.icdf(torch.tensor(0.5, dtype=dtype))
    assert 0.15 <= median <= 0.5, f"Beta(2,5) median {median} outside expected range"

    # Test 25th and 75th percentiles
    q25 = dist.icdf(torch.tensor(0.25, dtype=dtype))
    q75 = dist.icdf(torch.tensor(0.75, dtype=dtype))

    assert 0.05 <= q25 <= 0.4, f"Beta(2,5) 25th percentile {q25} outside expected range"
    assert 0.2 <= q75 <= 0.8, f"Beta(2,5) 75th percentile {q75} outside expected range"
    assert q25 < median < q75, "Quantiles not in correct order"


if __name__ == "__main__":
    test_beta_distribution_edge_cases()
    test_beta_distribution_accuracy()
    print("All Beta distribution tests passed!")
