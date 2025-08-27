import torch
from hypothesis import given
from hypothesis import strategies as st

import beignet.distributions


class TestStandardNormal:
    """Test suite for StandardNormal distribution."""

    def test_basic_properties(self):
        """Test basic properties of standard normal distribution."""
        dist = beignet.distributions.StandardNormal()

        # Test parameters
        assert torch.allclose(dist.mean, torch.tensor(0.0))
        assert torch.allclose(dist.variance, torch.tensor(1.0))
        assert torch.allclose(dist.stddev, torch.tensor(1.0))

        # Default dtype should be float32
        assert dist.dtype == torch.float32

    def test_from_dtype_constructor(self):
        """Test from_dtype constructor."""
        # Test float64
        dist_f64 = beignet.distributions.StandardNormal.from_dtype(torch.float64)
        assert dist_f64.dtype == torch.float64
        assert torch.allclose(dist_f64.mean, torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(dist_f64.variance, torch.tensor(1.0, dtype=torch.float64))

        # Test float32
        dist_f32 = beignet.distributions.StandardNormal.from_dtype(torch.float32)
        assert dist_f32.dtype == torch.float32
        assert torch.allclose(dist_f32.mean, torch.tensor(0.0, dtype=torch.float32))

    @given(
        prob=st.floats(min_value=0.01, max_value=0.99),
    )
    def test_icdf_basic_properties(self, prob):
        """Test basic properties of inverse CDF."""
        dist = beignet.distributions.StandardNormal()
        prob_tensor = torch.tensor(prob)

        quantile = dist.icdf(prob_tensor)

        # Basic properties
        assert torch.isfinite(quantile)
        assert quantile.dtype == torch.float32

        # Round-trip test: icdf(cdf(x)) ≈ x for reasonable x values
        x_test = torch.tensor(1.0)
        prob_test = dist.cdf(x_test)
        x_recovered = dist.icdf(prob_test)
        assert torch.allclose(x_recovered, x_test, rtol=1e-4, atol=1e-4)

    def test_common_critical_values(self):
        """Test commonly used critical values."""
        dist = beignet.distributions.StandardNormal()

        # Two-tailed critical values
        z_95_two = dist.icdf(torch.tensor(0.975))  # α = 0.05, two-tailed
        z_99_two = dist.icdf(torch.tensor(0.995))  # α = 0.01, two-tailed

        # One-tailed critical values
        z_95_one = dist.icdf(torch.tensor(0.95))  # α = 0.05, one-tailed
        z_99_one = dist.icdf(torch.tensor(0.99))  # α = 0.01, one-tailed

        # Check known values (with tolerance for numerical precision)
        assert torch.allclose(z_95_two, torch.tensor(1.96), rtol=1e-2)
        assert torch.allclose(z_99_two, torch.tensor(2.576), rtol=1e-2)
        assert torch.allclose(z_95_one, torch.tensor(1.645), rtol=1e-2)
        assert torch.allclose(z_99_one, torch.tensor(2.326), rtol=1e-2)

    def test_cdf_properties(self):
        """Test CDF properties."""
        dist = beignet.distributions.StandardNormal()

        # Test known values
        cdf_0 = dist.cdf(torch.tensor(0.0))
        cdf_neg_inf = dist.cdf(torch.tensor(-5.0))  # Approximately 0
        cdf_pos_inf = dist.cdf(torch.tensor(5.0))  # Approximately 1

        assert torch.allclose(cdf_0, torch.tensor(0.5), rtol=1e-4)
        assert cdf_neg_inf < 0.01
        assert cdf_pos_inf > 0.99

        # Test monotonicity
        x_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf_values = dist.cdf(x_values)

        for i in range(len(cdf_values) - 1):
            assert cdf_values[i + 1] > cdf_values[i]

    def test_icdf_monotonicity(self):
        """Test that icdf is monotonically increasing."""
        dist = beignet.distributions.StandardNormal()

        probs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles = dist.icdf(probs)

        # Should be monotonically increasing
        for i in range(len(quantiles) - 1):
            assert quantiles[i + 1] > quantiles[i]

    def test_equivalence_to_normal_0_1(self):
        """Test equivalence to Normal(0, 1)."""
        std_normal = beignet.distributions.StandardNormal()
        normal_0_1 = beignet.distributions.Normal(
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        # Test same quantiles
        probs = torch.tensor([0.1, 0.5, 0.9])
        quantiles_std = std_normal.icdf(probs)
        quantiles_norm = normal_0_1.icdf(probs)

        assert torch.allclose(quantiles_std, quantiles_norm, rtol=1e-5)

        # Test same CDF
        x_values = torch.tensor([-1.0, 0.0, 1.0])
        cdf_std = std_normal.cdf(x_values)
        cdf_norm = normal_0_1.cdf(x_values)

        assert torch.allclose(cdf_std, cdf_norm, rtol=1e-5)

    def test_sampling(self):
        """Test sampling from the distribution."""
        dist = beignet.distributions.StandardNormal()

        # Test basic sampling
        samples = dist.sample((1000,))
        assert samples.shape == (1000,)
        assert samples.dtype == torch.float32

        # Statistical tests (with tolerance for random variation)
        sample_mean = torch.mean(samples)
        sample_std = torch.std(samples, unbiased=True)

        # Should be approximately N(0, 1)
        assert torch.abs(sample_mean) < 0.1  # Mean ≈ 0
        assert torch.abs(sample_std - 1.0) < 0.1  # Std ≈ 1

    def test_log_prob(self):
        """Test log probability density function."""
        dist = beignet.distributions.StandardNormal()

        # Test at mean (should be maximum log prob)
        log_prob_0 = dist.log_prob(torch.tensor(0.0))
        log_prob_1 = dist.log_prob(torch.tensor(1.0))

        # At mean, log prob should be higher than at x=1
        assert log_prob_0 > log_prob_1

        # Should be finite
        assert torch.isfinite(log_prob_0)
        assert torch.isfinite(log_prob_1)

    def test_torch_compile_compatibility(self):
        """Test torch.compile compatibility."""
        dist = beignet.distributions.StandardNormal()
        prob = torch.tensor(0.75)

        # Test icdf compilation
        compiled_icdf = torch.compile(dist.icdf, fullgraph=True)
        result_compiled = compiled_icdf(prob)
        result_regular = dist.icdf(prob)

        assert torch.allclose(result_compiled, result_regular, rtol=1e-5)

        # Test cdf compilation
        x = torch.tensor(1.0)
        compiled_cdf = torch.compile(dist.cdf, fullgraph=True)
        cdf_compiled = compiled_cdf(x)
        cdf_regular = dist.cdf(x)

        assert torch.allclose(cdf_compiled, cdf_regular, rtol=1e-5)

    def test_gradient_computation(self):
        """Test gradient computation."""
        # Test icdf gradients
        prob = torch.tensor(0.5, requires_grad=True)
        dist = beignet.distributions.StandardNormal()

        quantile = dist.icdf(prob)
        loss = quantile.sum()
        loss.backward()

        assert prob.grad is not None
        assert torch.isfinite(prob.grad)

        # Test cdf gradients
        x = torch.tensor(1.0, requires_grad=True)
        cdf_val = dist.cdf(x)
        loss_cdf = cdf_val.sum()
        loss_cdf.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad)

    @given(
        batch_size=st.integers(min_value=1, max_value=5),
    )
    def test_batched_operations(self, batch_size):
        """Test batched operations."""
        dist = beignet.distributions.StandardNormal()

        # Test batched icdf
        probs = torch.rand(batch_size) * 0.8 + 0.1  # [0.1, 0.9]
        quantiles = dist.icdf(probs)

        assert quantiles.shape == (batch_size,)
        assert torch.all(torch.isfinite(quantiles))

        # Test batched cdf
        x_values = torch.randn(batch_size)
        cdf_values = dist.cdf(x_values)

        assert cdf_values.shape == (batch_size,)
        assert torch.all(cdf_values >= 0.0)
        assert torch.all(cdf_values <= 1.0)

    def test_repr(self):
        """Test string representation."""
        dist = beignet.distributions.StandardNormal()
        repr_str = repr(dist)
        assert "StandardNormal" in repr_str
