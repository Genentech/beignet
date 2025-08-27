import torch
from hypothesis import given
from hypothesis import strategies as st

import beignet.distributions


class TestPoisson:
    """Test suite for Poisson distribution."""

    @given(
        rate=st.floats(min_value=0.1, max_value=50.0),
        dtype=st.sampled_from([torch.float32, torch.float64]),
    )
    def test_poisson_distribution(self, rate, dtype):
        """Test Poisson distribution functionality."""
        rate_tensor = torch.tensor(rate, dtype=dtype)

        # Create distribution
        dist = beignet.distributions.Poisson(rate=rate_tensor)

        # Test basic properties
        assert dist.rate.dtype == dtype
        assert torch.allclose(dist.mean, rate_tensor, rtol=1e-4)
        assert torch.allclose(dist.variance, rate_tensor, rtol=1e-4)

        # Test sampling
        samples = dist.sample((100,))
        assert samples.dtype in [torch.float32, torch.float64]
        assert torch.all(samples >= 0)

        # Test log_prob with integer values (Poisson requires integers)
        test_values = torch.tensor(
            [0, 1, 2, int(rate), int(rate * 2)],
            dtype=torch.float32,
        )
        log_probs = dist.log_prob(test_values)
        assert log_probs.shape == test_values.shape
        assert torch.all(torch.isfinite(log_probs))

        # Skip CDF test as PyTorch's Poisson doesn't implement it

    @given(
        rate=st.floats(min_value=0.5, max_value=20.0),
        prob=st.floats(min_value=0.01, max_value=0.99),
        dtype=st.sampled_from([torch.float32, torch.float64]),
    )
    def test_icdf_basic_properties(self, rate, prob, dtype):
        """Test basic properties of inverse CDF."""
        rate_tensor = torch.tensor(rate, dtype=dtype)
        prob_tensor = torch.tensor(prob, dtype=dtype)

        dist = beignet.distributions.Poisson(rate=rate_tensor)

        # Test icdf
        quantile = dist.icdf(prob_tensor)

        # Basic properties
        assert quantile.dtype == dtype
        assert torch.all(quantile >= 0.0)
        assert torch.all(torch.isfinite(quantile))

        # Quantiles should be reasonable for the given rate
        if rate < 50:  # For reasonable rates
            assert quantile <= rate + 6 * torch.sqrt(
                rate_tensor,
            )  # Within 6 standard deviations

    def test_icdf_edge_cases(self):
        """Test edge cases for inverse CDF."""
        rate = torch.tensor(5.0)
        dist = beignet.distributions.Poisson(rate=rate)

        # Test edge probabilities
        q_low = dist.icdf(torch.tensor(1e-6))
        q_high = dist.icdf(torch.tensor(1 - 1e-6))

        assert q_low >= 0.0
        assert q_high >= q_low
        assert torch.isfinite(q_low)
        assert torch.isfinite(q_high)

        # Test probability = 0.5 (median)
        median = dist.icdf(torch.tensor(0.5))
        assert torch.isfinite(median)
        assert median >= 0.0

    def test_icdf_monotonicity(self):
        """Test that icdf is monotonically increasing."""
        rate = torch.tensor(10.0)
        dist = beignet.distributions.Poisson(rate=rate)

        probs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles = dist.icdf(probs)

        # Should be monotonically non-decreasing
        for i in range(len(quantiles) - 1):
            assert quantiles[i + 1] >= quantiles[i]

    @given(
        batch_size=st.integers(min_value=1, max_value=5),
        dtype=st.sampled_from([torch.float32, torch.float64]),
    )
    def test_batched_operations(self, batch_size, dtype):
        """Test batched operations."""
        rates = torch.rand(batch_size, dtype=dtype) * 20 + 0.5  # rates in [0.5, 20.5]
        probs = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.1  # probs in [0.1, 0.9]

        dist = beignet.distributions.Poisson(rate=rates)

        # Test batched icdf
        quantiles = dist.icdf(probs)
        assert quantiles.shape == (batch_size,)
        assert torch.all(quantiles >= 0.0)
        assert torch.all(torch.isfinite(quantiles))

    def test_different_rate_ranges(self):
        """Test icdf performance across different rate ranges."""
        # Small rates (≤ 10)
        small_rates = torch.tensor([0.5, 2.0, 5.0, 10.0])
        small_dist = beignet.distributions.Poisson(rate=small_rates)
        small_quantiles = small_dist.icdf(torch.tensor(0.9))
        assert torch.all(small_quantiles >= 0.0)
        assert torch.all(torch.isfinite(small_quantiles))

        # Medium rates (10 < λ ≤ 30)
        medium_rates = torch.tensor([15.0, 20.0, 25.0, 30.0])
        medium_dist = beignet.distributions.Poisson(rate=medium_rates)
        medium_quantiles = medium_dist.icdf(torch.tensor(0.9))
        assert torch.all(medium_quantiles >= 0.0)
        assert torch.all(torch.isfinite(medium_quantiles))

        # Large rates (> 30)
        large_rates = torch.tensor([35.0, 50.0, 100.0])
        large_dist = beignet.distributions.Poisson(rate=large_rates)
        large_quantiles = large_dist.icdf(torch.tensor(0.9))
        assert torch.all(large_quantiles >= 0.0)
        assert torch.all(torch.isfinite(large_quantiles))

    def test_torch_compile_compatibility(self):
        """Test torch.compile compatibility."""
        rate = torch.tensor(5.0)
        prob = torch.tensor(0.75)

        dist = beignet.distributions.Poisson(rate=rate)

        # Test icdf compilation
        compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

        # Should not raise compilation errors
        result_compiled = compiled_icdf(prob)
        result_regular = dist.icdf(prob)

        # Results should be close (allowing for numerical differences)
        assert torch.allclose(result_compiled, result_regular, rtol=1e-4, atol=1e-4)

    def test_gradient_computation(self):
        """Test gradient computation through icdf."""
        rate = torch.tensor(5.0, requires_grad=True)
        prob = torch.tensor(0.5)

        dist = beignet.distributions.Poisson(rate=rate)
        quantile = dist.icdf(prob)

        # Should be able to compute gradients
        loss = quantile.sum()
        loss.backward()

        assert rate.grad is not None
        assert torch.isfinite(rate.grad)

    def test_approximation_accuracy(self):
        """Test accuracy of different approximation methods."""
        # Test that our approximations give reasonable results
        # compared to the true quantiles for known cases

        # For Poisson(5), the 0.95 quantile should be around 9-10
        dist = beignet.distributions.Poisson(rate=torch.tensor(5.0))
        q95 = dist.icdf(torch.tensor(0.95))
        assert 7.0 <= q95 <= 12.0  # Reasonable range

        # For Poisson(1), the 0.9 quantile should be around 3-4
        dist_small = beignet.distributions.Poisson(rate=torch.tensor(1.0))
        q90_small = dist_small.icdf(torch.tensor(0.9))
        assert 2.0 <= q90_small <= 5.0  # Reasonable range

        # For Poisson(50), the 0.5 quantile should be close to 50
        dist_large = beignet.distributions.Poisson(rate=torch.tensor(50.0))
        median_large = dist_large.icdf(torch.tensor(0.5))
        assert 45.0 <= median_large <= 55.0  # Should be close to rate for large λ
