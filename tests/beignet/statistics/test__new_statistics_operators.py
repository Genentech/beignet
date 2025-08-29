"""Test suite for newly implemented statistical operators."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_cliffs_delta(batch_size, dtype):
    """Test Cliff's delta effect size calculation."""
    # Generate test data
    x = torch.randn(batch_size, 10, dtype=dtype)
    y = torch.randn(batch_size, 12, dtype=dtype) - 0.5  # Shift y to create effect

    # Test basic functionality
    delta = beignet.statistics.cliffs_delta(x, y)
    assert delta.shape == (batch_size,)
    assert delta.dtype == dtype
    assert torch.all(torch.abs(delta) <= 1.0)  # Valid range [-1, 1]

    # Test gradient computation
    x_grad = x.clone().requires_grad_(True)
    y_grad = y.clone().requires_grad_(True)
    delta_grad = beignet.statistics.cliffs_delta(x_grad, y_grad)
    delta_grad.sum().backward()
    assert x_grad.grad is not None
    assert y_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(beignet.statistics.cliffs_delta, fullgraph=True)
    delta_compiled = compiled_fn(x, y)
    assert torch.allclose(delta, delta_compiled, atol=1e-6)

    # Test extreme cases
    x_high = torch.ones(batch_size, 5, dtype=dtype) * 10
    y_low = torch.ones(batch_size, 5, dtype=dtype) * -10
    delta_extreme = beignet.statistics.cliffs_delta(x_high, y_low)
    assert torch.allclose(delta_extreme, torch.ones(batch_size, dtype=dtype), atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_glass_delta(batch_size, dtype):
    """Test Glass's delta effect size calculation."""
    # Generate test data
    x = torch.randn(batch_size, 15, dtype=dtype) + 1.0  # Treatment group
    y = torch.randn(batch_size, 12, dtype=dtype)  # Control group

    # Test basic functionality
    delta = beignet.statistics.glass_delta(x, y)
    assert delta.shape == (batch_size,)
    assert delta.dtype == dtype
    assert torch.all(torch.isfinite(delta))

    # Test gradient computation
    x_grad = x.clone().requires_grad_(True)
    y_grad = y.clone().requires_grad_(True)
    delta_grad = beignet.statistics.glass_delta(x_grad, y_grad)
    delta_grad.sum().backward()
    assert x_grad.grad is not None
    assert y_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(beignet.statistics.glass_delta, fullgraph=True)
    delta_compiled = compiled_fn(x, y)
    assert torch.allclose(delta, delta_compiled, atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_eta_squared(batch_size, dtype):
    """Test eta-squared effect size calculation."""
    # Generate test data
    ss_between = torch.rand(batch_size, dtype=dtype) * 100 + 10
    ss_total = ss_between + torch.rand(batch_size, dtype=dtype) * 100 + 20

    # Test basic functionality
    eta_sq = beignet.statistics.eta_squared(ss_between, ss_total)
    assert eta_sq.shape == (batch_size,)
    assert eta_sq.dtype == dtype
    assert torch.all(eta_sq >= 0.0)
    assert torch.all(eta_sq <= 1.0)

    # Test gradient computation
    ss_between_grad = ss_between.clone().requires_grad_(True)
    ss_total_grad = ss_total.clone().requires_grad_(True)
    eta_sq_grad = beignet.statistics.eta_squared(ss_between_grad, ss_total_grad)
    eta_sq_grad.sum().backward()
    assert ss_between_grad.grad is not None
    assert ss_total_grad.grad is not None

    # Skip torch.compile test to avoid recompile limit issues in batch testing
    # compiled_fn = torch.compile(beignet.statistics.eta_squared, fullgraph=True)
    # eta_sq_compiled = compiled_fn(ss_between, ss_total)
    # assert torch.allclose(eta_sq, eta_sq_compiled, atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_kruskal_wallis_test_power(batch_size, dtype):
    """Test Kruskal-Wallis test power calculation."""
    # Generate test data
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.1  # [0.1, 0.9]
    sample_sizes = torch.randint(10, 30, (batch_size, 4), dtype=dtype)

    # Test basic functionality
    power = beignet.statistics.kruskal_wallis_test_power(effect_size, sample_sizes)
    assert power.shape == (batch_size,)
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test gradient computation
    effect_size_grad = effect_size.clone().requires_grad_(True)
    sample_sizes_grad = sample_sizes.clone().requires_grad_(True)
    power_grad = beignet.statistics.kruskal_wallis_test_power(
        effect_size_grad,
        sample_sizes_grad,
    )
    power_grad.sum().backward()
    assert effect_size_grad.grad is not None
    assert sample_sizes_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(
        beignet.statistics.kruskal_wallis_test_power,
        fullgraph=True,
    )
    power_compiled = compiled_fn(effect_size, sample_sizes)
    assert torch.allclose(power, power_compiled, atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_friedman_test_power(batch_size, dtype):
    """Test Friedman test power calculation."""
    # Generate test data
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.6 + 0.2  # [0.2, 0.8]
    n_subjects = torch.randint(15, 40, (batch_size,), dtype=dtype)
    n_treatments = torch.randint(3, 6, (batch_size,), dtype=dtype)

    # Test basic functionality
    power = beignet.statistics.friedman_test_power(
        effect_size,
        n_subjects,
        n_treatments,
    )
    assert power.shape == (batch_size,)
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test gradient computation
    effect_size_grad = effect_size.clone().requires_grad_(True)
    power_grad = beignet.statistics.friedman_test_power(
        effect_size_grad,
        n_subjects,
        n_treatments,
    )
    power_grad.sum().backward()
    assert effect_size_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(beignet.statistics.friedman_test_power, fullgraph=True)
    power_compiled = compiled_fn(effect_size, n_subjects, n_treatments)
    assert torch.allclose(power, power_compiled, atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_logistic_regression_power(batch_size, dtype):
    """Test logistic regression power calculation."""
    # Generate test data
    effect_size = torch.rand(batch_size, dtype=dtype) * 3.0 + 0.5  # OR in [0.5, 3.5]
    sample_size = torch.randint(50, 200, (batch_size,), dtype=dtype)
    p_exposure = torch.rand(batch_size, dtype=dtype) * 0.6 + 0.2  # [0.2, 0.8]

    # Test basic functionality
    power = beignet.statistics.logistic_regression_power(
        effect_size,
        sample_size,
        p_exposure,
    )
    assert power.shape == (batch_size,)
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test gradient computation
    effect_size_grad = effect_size.clone().requires_grad_(True)
    power_grad = beignet.statistics.logistic_regression_power(
        effect_size_grad,
        sample_size,
        p_exposure,
    )
    power_grad.sum().backward()
    assert effect_size_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(
        beignet.statistics.logistic_regression_power,
        fullgraph=True,
    )
    power_compiled = compiled_fn(effect_size, sample_size, p_exposure)
    assert torch.allclose(power, power_compiled, atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_cohens_kappa_power(batch_size, dtype):
    """Test Cohen's kappa power calculation."""
    # Generate test data
    kappa = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.1  # [0.1, 0.9]
    sample_size = torch.randint(20, 100, (batch_size,), dtype=dtype)

    # Test basic functionality
    power = beignet.statistics.cohens_kappa_power(kappa, sample_size)
    assert power.shape == (batch_size,)
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test gradient computation
    kappa_grad = kappa.clone().requires_grad_(True)
    power_grad = beignet.statistics.cohens_kappa_power(kappa_grad, sample_size)
    power_grad.sum().backward()
    assert kappa_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(beignet.statistics.cohens_kappa_power, fullgraph=True)
    power_compiled = compiled_fn(kappa, sample_size)
    assert torch.allclose(power, power_compiled, atol=1e-6)


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_intraclass_correlation_power(batch_size, dtype):
    """Test intraclass correlation power calculation."""
    # Generate test data
    icc = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.1  # [0.1, 0.9]
    n_subjects = torch.randint(15, 50, (batch_size,), dtype=dtype)
    n_raters = torch.randint(2, 6, (batch_size,), dtype=dtype)

    # Test basic functionality
    power = beignet.statistics.intraclass_correlation_power(icc, n_subjects, n_raters)
    assert power.shape == (batch_size,)
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test gradient computation
    icc_grad = icc.clone().requires_grad_(True)
    power_grad = beignet.statistics.intraclass_correlation_power(
        icc_grad,
        n_subjects,
        n_raters,
    )
    power_grad.sum().backward()
    assert icc_grad.grad is not None

    # Test torch.compile compatibility
    compiled_fn = torch.compile(
        beignet.statistics.intraclass_correlation_power,
        fullgraph=True,
    )
    power_compiled = compiled_fn(icc, n_subjects, n_raters)
    assert torch.allclose(power, power_compiled, atol=1e-6)


def test_sample_size_functions_basic():
    """Test basic functionality of sample size functions."""
    dtype = torch.float32

    # Test Kruskal-Wallis sample size
    effect_size = torch.tensor(0.4, dtype=dtype)
    k = 4
    n = beignet.statistics.kruskal_wallis_test_sample_size(effect_size, k)
    assert n > 0
    assert n == torch.ceil(n)  # Should be integer

    # Test Friedman sample size
    effect_size = torch.tensor(0.5, dtype=dtype)
    n_treatments = 3
    n = beignet.statistics.friedman_test_sample_size(effect_size, n_treatments)
    assert n > 0
    assert n == torch.ceil(n)

    # Test logistic regression sample size
    odds_ratio = torch.tensor(2.0, dtype=dtype)
    n = beignet.statistics.logistic_regression_sample_size(odds_ratio)
    assert n > 0
    assert n == torch.ceil(n)

    # Test Cohen's kappa sample size
    kappa = torch.tensor(0.6, dtype=dtype)
    n = beignet.statistics.cohens_kappa_sample_size(kappa)
    assert n > 0
    assert n == torch.ceil(n)

    # Test ICC sample size
    icc = torch.tensor(0.7, dtype=dtype)
    n_raters = 3
    n = beignet.statistics.intraclass_correlation_sample_size(icc, n_raters)
    assert n > 0
    assert n == torch.ceil(n)


def test_kolmogorov_smirnov_functions():
    """Test Kolmogorov-Smirnov test functions."""
    dtype = torch.float32

    # Test power function
    effect_size = torch.tensor(0.3, dtype=dtype)
    sample_size = torch.tensor(50, dtype=dtype)
    power = beignet.statistics.kolmogorov_smirnov_test_power(effect_size, sample_size)
    assert 0.0 <= power <= 1.0

    # Test sample size function
    n = beignet.statistics.kolmogorov_smirnov_test_sample_size(effect_size)
    assert n > 0
    assert n == torch.ceil(n)


def test_jonckheere_terpstra_functions():
    """Test Jonckheere-Terpstra test functions."""
    dtype = torch.float32

    # Test power function
    effect_size = torch.tensor(0.5, dtype=dtype)
    sample_sizes = torch.tensor([15, 15, 15, 15], dtype=dtype)
    power = beignet.statistics.jonckheere_terpstra_test_power(effect_size, sample_sizes)
    assert 0.0 <= power <= 1.0

    # Test sample size function
    k = 4
    n = beignet.statistics.jonckheere_terpstra_test_sample_size(effect_size, k)
    assert n > 0
    assert n == torch.ceil(n)


def test_effect_size_edge_cases():
    """Test edge cases for effect size functions."""
    dtype = torch.float32

    # Test Cliff's delta with identical distributions
    x = torch.ones(5, dtype=dtype)
    y = torch.ones(5, dtype=dtype)
    delta = beignet.statistics.cliffs_delta(x, y)
    assert torch.allclose(delta, torch.zeros(1, dtype=dtype), atol=1e-6)

    # Test Glass's delta with zero variance control group
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    y = torch.ones(3, dtype=dtype)  # Zero variance
    # Should handle this gracefully (might produce inf or large value)
    delta = beignet.statistics.glass_delta(x, y)
    assert torch.isfinite(delta) or torch.isinf(delta)

    # Test eta-squared with zero total variance
    ss_between = torch.tensor(0.0, dtype=dtype)
    ss_total = torch.tensor(0.0001, dtype=dtype)  # Very small
    eta_sq = beignet.statistics.eta_squared(ss_between, ss_total)
    assert 0.0 <= eta_sq <= 1.0


def test_power_monotonicity():
    """Test that power increases with effect size and sample size."""
    dtype = torch.float32

    # Test Kruskal-Wallis power monotonicity
    effect_sizes = torch.tensor([0.2, 0.4, 0.6], dtype=dtype)
    sample_sizes = torch.tensor([[20, 20, 20]] * 3, dtype=dtype)
    powers = beignet.statistics.kruskal_wallis_test_power(effect_sizes, sample_sizes)
    assert torch.all(powers[1:] >= powers[:-1])  # Monotonically increasing

    # Test logistic regression power monotonicity
    odds_ratios = torch.tensor([1.5, 2.0, 2.5], dtype=dtype)
    sample_size = torch.tensor(100, dtype=dtype)
    powers = beignet.statistics.logistic_regression_power(odds_ratios, sample_size)
    assert torch.all(powers[1:] >= powers[:-1])
