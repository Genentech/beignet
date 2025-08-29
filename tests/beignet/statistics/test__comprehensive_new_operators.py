"""Comprehensive test suite for all newly implemented statistical operators."""

import torch

import beignet.statistics as bs


def test_all_operators_basic_functionality():
    """Test that all operators work with basic inputs."""
    dtype = torch.float32

    # Effect size operators
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    y = torch.tensor([4.0, 5.0, 6.0], dtype=dtype)
    assert bs.cliffs_delta(x, y) is not None
    assert bs.glass_delta(x, y) is not None

    ss_between = torch.tensor(30.0, dtype=dtype)
    ss_total = torch.tensor(100.0, dtype=dtype)
    assert bs.eta_squared(ss_between, ss_total) is not None

    # Non-parametric tests
    effect = torch.tensor(0.3, dtype=dtype)
    sample_sizes = torch.tensor([15, 15, 15], dtype=dtype)
    assert bs.kruskal_wallis_test_power(effect, sample_sizes) is not None
    assert bs.kruskal_wallis_test_sample_size(effect, 3) is not None

    n_subj = torch.tensor(20, dtype=dtype)
    n_treat = torch.tensor(3, dtype=dtype)
    assert bs.friedman_test_power(effect, n_subj, n_treat) is not None
    assert bs.friedman_test_sample_size(effect, n_treat) is not None

    assert bs.jonckheere_terpstra_test_power(effect, sample_sizes) is not None
    assert bs.jonckheere_terpstra_test_sample_size(effect, 3) is not None

    n_obs = torch.tensor(50, dtype=dtype)
    assert bs.kolmogorov_smirnov_test_power(effect, n_obs) is not None
    assert bs.kolmogorov_smirnov_test_sample_size(effect) is not None

    # Regression operators
    or_val = torch.tensor(2.0, dtype=dtype)
    n_sample = torch.tensor(100, dtype=dtype)
    assert bs.logistic_regression_power(or_val, n_sample) is not None
    assert bs.logistic_regression_sample_size(or_val) is not None

    r2 = torch.tensor(0.15, dtype=dtype)
    n_pred = torch.tensor(3, dtype=dtype)
    assert bs.multivariable_linear_regression_power(r2, n_sample, n_pred) is not None
    assert bs.multivariable_linear_regression_sample_size(r2, n_pred) is not None

    irr = torch.tensor(1.5, dtype=dtype)
    mean_rate = torch.tensor(2.0, dtype=dtype)
    assert bs.poisson_regression_power(irr, n_sample, mean_rate) is not None
    assert bs.poisson_regression_sample_size(irr, mean_rate) is not None

    # ANOVA extensions
    n_per_cell = torch.tensor(20, dtype=dtype)
    a_levels = torch.tensor(2, dtype=dtype)
    b_levels = torch.tensor(3, dtype=dtype)
    assert (
        bs.two_way_analysis_of_variance_power(effect, n_per_cell, a_levels, b_levels)
        is not None
    )
    assert (
        bs.two_way_analysis_of_variance_sample_size(effect, a_levels, b_levels)
        is not None
    )

    # Advanced operators
    kappa = torch.tensor(0.6, dtype=dtype)
    assert bs.cohens_kappa_power(kappa, n_obs) is not None
    assert bs.cohens_kappa_sample_size(kappa) is not None

    icc = torch.tensor(0.7, dtype=dtype)
    n_raters = torch.tensor(3, dtype=dtype)
    assert bs.intraclass_correlation_power(icc, n_subj, n_raters) is not None
    assert bs.intraclass_correlation_sample_size(icc, n_raters) is not None

    obs_per_subj = torch.tensor(4, dtype=dtype)
    icc_mixed = torch.tensor(0.1, dtype=dtype)
    assert bs.mixed_model_power(effect, n_subj, obs_per_subj, icc_mixed) is not None
    assert bs.mixed_model_sample_size(effect, obs_per_subj, icc_mixed) is not None

    n_time = torch.tensor(50, dtype=dtype)
    n_pre = torch.tensor(25, dtype=dtype)
    assert bs.interrupted_time_series_power(effect, n_time, n_pre) is not None
    assert bs.interrupted_time_series_sample_size(effect) is not None


def test_operators_return_valid_ranges():
    """Test that all power functions return values in [0,1] and sample sizes are positive."""
    dtype = torch.float64

    # Power functions should return [0,1]
    power_funcs = [
        (
            bs.kruskal_wallis_test_power,
            [torch.tensor(0.3, dtype=dtype), torch.tensor([15, 15, 15], dtype=dtype)],
        ),
        (
            bs.friedman_test_power,
            [
                torch.tensor(0.4, dtype=dtype),
                torch.tensor(20, dtype=dtype),
                torch.tensor(3, dtype=dtype),
            ],
        ),
        (
            bs.logistic_regression_power,
            [torch.tensor(2.0, dtype=dtype), torch.tensor(100, dtype=dtype)],
        ),
        (
            bs.multivariable_linear_regression_power,
            [
                torch.tensor(0.15, dtype=dtype),
                torch.tensor(100, dtype=dtype),
                torch.tensor(3, dtype=dtype),
            ],
        ),
        (
            bs.two_way_analysis_of_variance_power,
            [
                torch.tensor(0.25, dtype=dtype),
                torch.tensor(20, dtype=dtype),
                torch.tensor(2, dtype=dtype),
                torch.tensor(3, dtype=dtype),
            ],
        ),
        (
            bs.cohens_kappa_power,
            [torch.tensor(0.6, dtype=dtype), torch.tensor(50, dtype=dtype)],
        ),
        (
            bs.mixed_model_power,
            [
                torch.tensor(0.5, dtype=dtype),
                torch.tensor(30, dtype=dtype),
                torch.tensor(4, dtype=dtype),
                torch.tensor(0.1, dtype=dtype),
            ],
        ),
    ]

    for func, args in power_funcs:
        result = func(*args)
        assert torch.all(result >= 0.0), f"{func.__name__} returned negative power"
        assert torch.all(result <= 1.0), f"{func.__name__} returned power > 1"

    # Sample size functions should return positive integers
    sample_size_funcs = [
        (bs.kruskal_wallis_test_sample_size, [torch.tensor(0.3, dtype=dtype), 3]),
        (
            bs.friedman_test_sample_size,
            [torch.tensor(0.4, dtype=dtype), torch.tensor(3, dtype=dtype)],
        ),
        (bs.logistic_regression_sample_size, [torch.tensor(2.0, dtype=dtype)]),
        (
            bs.multivariable_linear_regression_sample_size,
            [torch.tensor(0.15, dtype=dtype), torch.tensor(3, dtype=dtype)],
        ),
        (bs.cohens_kappa_sample_size, [torch.tensor(0.6, dtype=dtype)]),
        (
            bs.mixed_model_sample_size,
            [
                torch.tensor(0.5, dtype=dtype),
                torch.tensor(4, dtype=dtype),
                torch.tensor(0.1, dtype=dtype),
            ],
        ),
    ]

    for func, args in sample_size_funcs:
        result = func(*args)
        assert torch.all(result > 0), (
            f"{func.__name__} returned non-positive sample size"
        )
        assert torch.all(result == torch.ceil(result)), (
            f"{func.__name__} returned non-integer sample size"
        )


def test_gradient_computation():
    """Test that key operators support gradient computation."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    # Test Cliff's delta gradients
    delta = bs.cliffs_delta(x, y)
    delta.backward()
    assert x.grad is not None
    assert y.grad is not None

    # Test effect size operators
    ss_between = torch.tensor(30.0, requires_grad=True)
    ss_total = torch.tensor(100.0, requires_grad=True)
    eta_sq = bs.eta_squared(ss_between, ss_total)
    eta_sq.backward()
    assert ss_between.grad is not None
    assert ss_total.grad is not None


def test_torch_compile_compatibility():
    """Test that operators work with torch.compile."""
    # Test a few key operators
    compiled_cliffs = torch.compile(bs.cliffs_delta, fullgraph=True)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result1 = bs.cliffs_delta(x, y)
    result2 = compiled_cliffs(x, y)
    assert torch.allclose(result1, result2)

    compiled_eta = torch.compile(bs.eta_squared, fullgraph=True)
    ss_between = torch.tensor(30.0)
    ss_total = torch.tensor(100.0)
    result1 = bs.eta_squared(ss_between, ss_total)
    result2 = compiled_eta(ss_between, ss_total)
    assert torch.allclose(result1, result2)


def test_batch_operations():
    """Test that operators work with batched inputs."""
    batch_size = 3
    dtype = torch.float32

    # Test batched Cliff's delta
    x = torch.randn(batch_size, 10, dtype=dtype)
    y = torch.randn(batch_size, 12, dtype=dtype)
    delta = bs.cliffs_delta(x, y)
    assert delta.shape == (batch_size,)

    # Test batched eta-squared
    ss_between = torch.rand(batch_size, dtype=dtype) * 50
    ss_total = ss_between + torch.rand(batch_size, dtype=dtype) * 50
    eta_sq = bs.eta_squared(ss_between, ss_total)
    assert eta_sq.shape == (batch_size,)

    # Test batched power calculation
    effect_sizes = torch.tensor([0.2, 0.4, 0.6], dtype=dtype)
    sample_sizes = torch.tensor([[15, 15, 15]] * batch_size, dtype=dtype)
    powers = bs.kruskal_wallis_test_power(effect_sizes, sample_sizes)
    assert powers.shape == (batch_size,)
    assert torch.all(
        powers[2] >= powers[1],
    )  # Higher effect size should give higher power
    assert torch.all(powers[1] >= powers[0])


if __name__ == "__main__":
    test_all_operators_basic_functionality()
    test_operators_return_valid_ranges()
    test_gradient_computation()
    test_torch_compile_compatibility()
    test_batch_operations()
    print("All comprehensive tests passed!")
