import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_analysis_of_covariance_power(batch_size, dtype):
    """Test Analysis of Covariance (ANCOVA) power calculation."""

    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([30, 60, 90], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    num_groups = torch.tensor([2, 3, 4], dtype=dtype).repeat(batch_size, 1).flatten()
    covariate_r2 = (
        torch.tensor([0.1, 0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    num_covariates = (
        torch.tensor([1, 2, 3], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.analysis_of_covariance_power(
        effect_sizes,
        sample_sizes,
        num_groups,
        covariate_r2,
        num_covariates,
    )
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.analysis_of_covariance_power(
        effect_sizes,
        sample_sizes,
        num_groups,
        covariate_r2,
        num_covariates,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(60.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.3, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    large_effect = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.8, dtype=dtype),
        torch.tensor(60.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.3, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.3, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    large_n = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(90.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.3, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test that power increases with covariate correlation
    low_r2 = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(60.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    high_r2 = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(60.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    assert high_r2 > low_r2

    # Test different alpha values
    power_05 = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(60.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.3, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
        alpha=0.05,
    )
    power_01 = beignet.statistics.analysis_of_covariance_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(60.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(0.3, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
        alpha=0.01,
    )
    assert power_05 > power_01

    # Test gradient computation
    effect_grad = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    sample_grad = torch.tensor([60.0], dtype=dtype, requires_grad=True)
    groups_grad = torch.tensor([3.0], dtype=dtype, requires_grad=True)
    cov_r2_grad = torch.tensor([0.3], dtype=dtype, requires_grad=True)
    num_cov_grad = torch.tensor([2.0], dtype=dtype, requires_grad=True)

    result_grad = beignet.statistics.analysis_of_covariance_power(
        effect_grad,
        sample_grad,
        groups_grad,
        cov_r2_grad,
        num_cov_grad,
    )

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None
    assert groups_grad.grad is not None
    assert cov_r2_grad.grad is not None
    assert num_cov_grad.grad is not None

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        beignet.statistics.analysis_of_covariance_power,
        fullgraph=True,
    )
    result_compiled = compiled_func(
        effect_sizes[:1],
        sample_sizes[:1],
        num_groups[:1],
        covariate_r2[:1],
        num_covariates[:1],
    )
    result_normal = beignet.statistics.analysis_of_covariance_power(
        effect_sizes[:1],
        sample_sizes[:1],
        num_groups[:1],
        covariate_r2[:1],
        num_covariates[:1],
    )
    assert torch.allclose(result_compiled, result_normal, atol=1e-6)
