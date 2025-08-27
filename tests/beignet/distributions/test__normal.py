import torch

import beignet.distributions


def test_normal_distribution():
    """Test Normal distribution functionality."""
    dtype = torch.float64

    # Test basic functionality
    loc = torch.tensor(0.0, dtype=dtype)
    scale = torch.tensor(1.0, dtype=dtype)

    dist = beignet.distributions.Normal(loc, scale)

    # Test that it inherits from torch.distributions.Normal
    assert isinstance(dist, torch.distributions.Normal)

    # Test icdf with standard normal quantiles
    probabilities = torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975], dtype=dtype)
    quantiles = dist.icdf(probabilities)

    # Test output shape and known values
    assert quantiles.shape == (5,)

    # Known standard normal quantiles (approximately)
    expected_quantiles = torch.tensor([-1.96, -1.28, 0.0, 1.28, 1.96], dtype=dtype)
    assert torch.allclose(quantiles, expected_quantiles, atol=1e-2)

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

    # Test with different parameters
    # Non-standard normal: N(5, 2)
    nonstandard_dist = beignet.distributions.Normal(
        torch.tensor(5.0, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )

    # Median should be at the mean
    median = nonstandard_dist.icdf(torch.tensor(0.5, dtype=dtype))
    assert torch.allclose(median, torch.tensor(5.0, dtype=dtype), atol=1e-10)

    # Test symmetry around mean
    lower_q = nonstandard_dist.icdf(torch.tensor(0.25, dtype=dtype))
    upper_q = nonstandard_dist.icdf(torch.tensor(0.75, dtype=dtype))
    mean_val = torch.tensor(5.0, dtype=dtype)

    # Distance from mean should be symmetric
    lower_dist = torch.abs(lower_q - mean_val)
    upper_dist = torch.abs(upper_q - mean_val)
    assert torch.allclose(lower_dist, upper_dist, atol=1e-10)

    # Test torch.compile compatibility
    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    test_prob = torch.tensor(0.3, dtype=dtype)
    original_result = dist.icdf(test_prob)
    compiled_result = compiled_icdf(test_prob)

    assert torch.allclose(original_result, compiled_result, rtol=1e-10)

    # Test gradients
    loc_grad = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    scale_grad = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    grad_dist = beignet.distributions.Normal(loc_grad, scale_grad)

    prob = torch.tensor(0.4, dtype=dtype)
    quantile = grad_dist.icdf(prob)

    # Test that gradients can be computed
    quantile.backward()

    assert loc_grad.grad is not None
    assert scale_grad.grad is not None
    assert torch.isfinite(loc_grad.grad)
    assert torch.isfinite(scale_grad.grad)


def test_normal_distribution_accuracy():
    """Test accuracy against known standard normal values."""
    dtype = torch.float64

    # Standard normal distribution
    std_normal = beignet.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    # Test common critical values
    test_cases = [
        (0.025, -1.959963984540054),  # 2.5th percentile
        (0.05, -1.6448536269514722),  # 5th percentile
        (0.1, -1.2815515655446004),  # 10th percentile
        (0.5, 0.0),  # 50th percentile (median)
        (0.9, 1.2815515655446004),  # 90th percentile
        (0.95, 1.6448536269514722),  # 95th percentile
        (0.975, 1.959963984540054),  # 97.5th percentile
    ]

    for prob, expected in test_cases:
        prob_tensor = torch.tensor(prob, dtype=dtype)
        quantile = std_normal.icdf(prob_tensor)
        expected_tensor = torch.tensor(expected, dtype=dtype)

        assert torch.allclose(quantile, expected_tensor, atol=1e-10), (
            f"Standard normal {prob} quantile: got {quantile}, expected {expected}"
        )


def test_normal_distribution_consistency():
    """Test consistency with torch.distributions.Normal."""
    dtype = torch.float64

    # Create distributions using both interfaces
    loc = torch.tensor(2.5, dtype=dtype)
    scale = torch.tensor(1.5, dtype=dtype)

    beignet_dist = beignet.distributions.Normal(loc, scale)
    torch_dist = torch.distributions.Normal(loc, scale)

    # Test that icdf results are identical
    test_probs = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], dtype=dtype)

    beignet_quantiles = beignet_dist.icdf(test_probs)
    torch_quantiles = torch_dist.icdf(test_probs)

    assert torch.allclose(beignet_quantiles, torch_quantiles, rtol=1e-15), (
        "beignet.distributions.Normal should match torch.distributions.Normal exactly"
    )

    # Test that other methods still work (inherited)
    test_values = torch.tensor([-1.0, 0.0, 2.5, 5.0, 7.0], dtype=dtype)

    beignet_pdf = beignet_dist.log_prob(test_values)
    torch_pdf = torch_dist.log_prob(test_values)

    assert torch.allclose(beignet_pdf, torch_pdf, rtol=1e-15)


def test_normal_distribution_batched():
    """Test Normal distribution with batched parameters."""
    dtype = torch.float64

    # Batched parameters
    locs = torch.tensor([0.0, 1.0, -2.0], dtype=dtype)
    scales = torch.tensor([1.0, 2.0, 0.5], dtype=dtype)

    dist = beignet.distributions.Normal(locs, scales)

    # Single probability, multiple distributions
    prob = torch.tensor(0.5, dtype=dtype)
    medians = dist.icdf(prob)

    # Medians should equal the means
    assert torch.allclose(medians, locs, atol=1e-10)

    # Multiple probabilities, multiple distributions
    probs = torch.tensor([0.25, 0.5, 0.75], dtype=dtype)
    quantiles = dist.icdf(probs)

    assert quantiles.shape == (3,)

    # Check that each quantile comes from the correct distribution
    for i in range(3):
        single_dist = beignet.distributions.Normal(locs[i], scales[i])
        expected = single_dist.icdf(probs[i])
        assert torch.allclose(quantiles[i], expected, atol=1e-10)


if __name__ == "__main__":
    test_normal_distribution()
    test_normal_distribution_accuracy()
    test_normal_distribution_consistency()
    test_normal_distribution_batched()
    print("All Normal distribution tests passed!")
