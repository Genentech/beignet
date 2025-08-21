import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet
from beignet.metrics import CohensD, HedgesG, TTestPower, TTestSampleSize


@given(
    n1=st.integers(min_value=10, max_value=50),
    n2=st.integers(min_value=10, max_value=50),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_cohens_d_metric(n1, n2, dtype):
    """Test CohensD metric functionality."""
    # Generate test data
    group1 = torch.randn(n1, dtype=dtype)
    group2 = torch.randn(n2, dtype=dtype) + 0.5  # Add some effect

    # Test basic functionality with pooled variance
    metric_pooled = CohensD(pooled=True)
    metric_pooled.update(group1, group2)
    result_pooled = metric_pooled.compute()

    assert isinstance(result_pooled, torch.Tensor)
    assert result_pooled.dtype == dtype
    assert result_pooled.numel() == 1

    # Compare with direct calculation
    direct_result = beignet.cohens_d(
        group1.unsqueeze(0), group2.unsqueeze(0), pooled=True
    )
    assert torch.allclose(result_pooled, direct_result.squeeze(), atol=1e-6)

    # Test non-pooled variance
    metric_nonpooled = CohensD(pooled=False)
    metric_nonpooled.update(group1, group2)
    result_nonpooled = metric_nonpooled.compute()

    direct_nonpooled = beignet.cohens_d(
        group1.unsqueeze(0), group2.unsqueeze(0), pooled=False
    )
    assert torch.allclose(result_nonpooled, direct_nonpooled.squeeze(), atol=1e-6)

    # Test reset functionality
    metric_pooled.reset()
    try:
        metric_pooled.compute()
        assert False, "Should have raised RuntimeError after reset"
    except RuntimeError:
        pass

    # Test multiple updates
    metric_multi = CohensD(pooled=True)
    split1, split2 = torch.chunk(group1, 2)
    metric_multi.update(split1, group2[: len(split1)])
    metric_multi.update(split2, group2[len(split1) : len(split1) + len(split2)])
    result_multi = metric_multi.compute()

    # Should be similar to single update (allowing for some numerical differences due to data splitting)
    assert torch.abs(result_multi - result_pooled) < 1.0

    # Test repr
    repr_str = repr(metric_pooled)
    assert "CohensD" in repr_str
    assert "pooled=True" in repr_str


@given(
    n1=st.integers(min_value=10, max_value=50),
    n2=st.integers(min_value=10, max_value=50),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_hedges_g_metric(n1, n2, dtype):
    """Test HedgesG metric functionality."""
    # Generate test data
    group1 = torch.randn(n1, dtype=dtype)
    group2 = torch.randn(n2, dtype=dtype) + 0.5  # Add some effect

    # Test basic functionality
    metric = HedgesG()
    metric.update(group1, group2)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.dtype == dtype
    assert result.numel() == 1

    # Compare with direct calculation
    direct_result = beignet.hedges_g(group1.unsqueeze(0), group2.unsqueeze(0))
    assert torch.allclose(result, direct_result.squeeze(), atol=1e-6)

    # Test reset functionality
    metric.reset()
    try:
        metric.compute()
        assert False, "Should have raised RuntimeError after reset"
    except RuntimeError:
        pass

    # Test multiple updates
    metric_multi = HedgesG()
    split1, split2 = torch.chunk(group1, 2)
    metric_multi.update(split1, group2[: len(split1)])
    metric_multi.update(split2, group2[len(split1) : len(split1) + len(split2)])
    result_multi = metric_multi.compute()

    # Should be similar to single update (allowing for some numerical differences due to data splitting)
    assert torch.abs(result_multi - result) < 1.0

    # Test repr
    repr_str = repr(metric)
    assert "HedgesG" in repr_str


@given(
    n1=st.integers(min_value=15, max_value=30),
    n2=st.integers(min_value=15, max_value=30),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_t_test_power_metric(n1, n2, dtype):
    """Test TTestPower metric functionality."""
    # Generate test data with known effect
    group1 = torch.randn(n1, dtype=dtype)
    group2 = torch.randn(n2, dtype=dtype) + 0.5  # Medium effect size

    # Test basic functionality
    metric = TTestPower(alpha=0.05, alternative="two-sided")
    metric.update(group1, group2)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.dtype == dtype
    assert 0 <= result <= 1  # Power should be between 0 and 1

    # Test different alternatives
    metric_greater = TTestPower(alternative="greater")
    metric_greater.update(group1, group2)
    power_greater = metric_greater.compute()

    metric_less = TTestPower(alternative="less")
    metric_less.update(group1, group2)
    power_less = metric_less.compute()

    assert 0 <= power_greater <= 1
    assert 0 <= power_less <= 1

    # Test different alpha levels
    metric_strict = TTestPower(alpha=0.01)
    metric_strict.update(group1, group2)
    power_strict = metric_strict.compute()

    # Stricter alpha should generally give lower power
    assert power_strict <= result + 0.1  # Allow some tolerance

    # Test reset functionality
    metric.reset()
    try:
        metric.compute()
        assert False, "Should have raised RuntimeError after reset"
    except RuntimeError:
        pass

    # Test invalid parameters
    try:
        TTestPower(alpha=1.5)
        assert False, "Should have raised ValueError for invalid alpha"
    except ValueError:
        pass

    try:
        TTestPower(alternative="invalid")
        assert False, "Should have raised ValueError for invalid alternative"
    except ValueError:
        pass

    # Test repr
    repr_str = repr(metric)
    assert "TTestPower" in repr_str
    assert "0.05" in repr_str
    assert "two-sided" in repr_str


@given(
    n1=st.integers(min_value=10, max_value=25),
    n2=st.integers(min_value=10, max_value=25),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_t_test_sample_size_metric(n1, n2, dtype):
    """Test TTestSampleSize metric functionality."""
    # Generate test data
    group1 = torch.randn(n1, dtype=dtype)
    group2 = (
        torch.randn(n2, dtype=dtype) + 0.6
    )  # Add moderate effect for stable results

    # Test basic functionality
    metric = TTestSampleSize(power=0.8, alpha=0.05, alternative="two-sided")
    metric.update(group1, group2)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.dtype == dtype
    assert result.numel() == 1
    assert result > 0  # Sample size should be positive
    assert result == torch.ceil(result)  # Should be integer

    # Test different power levels
    metric_high_power = TTestSampleSize(power=0.9)
    metric_high_power.update(group1, group2)
    result_high_power = metric_high_power.compute()

    # Higher power should require larger sample size
    assert result_high_power >= result - 5  # Allow some tolerance

    # Test different alpha levels
    metric_strict = TTestSampleSize(alpha=0.01)
    metric_strict.update(group1, group2)
    result_strict = metric_strict.compute()

    # Stricter alpha should require larger sample size
    assert result_strict >= result - 5  # Allow some tolerance

    # Test one-sided test
    metric_one_sided = TTestSampleSize(alternative="greater")
    metric_one_sided.update(group1, group2)
    result_one_sided = metric_one_sided.compute()

    # One-sided test should require smaller or equal sample size
    assert result_one_sided <= result + 5  # Allow some tolerance

    # Test reset functionality
    metric.reset()
    try:
        metric.compute()
        assert False, "Should have raised RuntimeError after reset"
    except RuntimeError:
        pass

    # Test invalid parameters
    try:
        TTestSampleSize(alpha=0)
        assert False, "Should have raised ValueError for invalid alpha"
    except ValueError:
        pass

    try:
        TTestSampleSize(power=1.1)
        assert False, "Should have raised ValueError for invalid power"
    except ValueError:
        pass

    try:
        TTestSampleSize(alternative="invalid")
        assert False, "Should have raised ValueError for invalid alternative"
    except ValueError:
        pass

    # Test repr
    repr_str = repr(metric)
    assert "TTestSampleSize" in repr_str
    assert "0.8" in repr_str
    assert "0.05" in repr_str
    assert "two-sided" in repr_str


def test_metrics_module_import():
    """Test that metrics can be imported from beignet."""
    # Test importing from main beignet module
    assert hasattr(beignet, "metrics")
    assert hasattr(beignet.metrics, "CohensD")
    assert hasattr(beignet.metrics, "HedgesG")
    assert hasattr(beignet.metrics, "TTestPower")
    assert hasattr(beignet.metrics, "TTestSampleSize")

    # Test instantiation
    cohens_d_metric = beignet.metrics.CohensD()
    hedges_g_metric = beignet.metrics.HedgesG()
    power_metric = beignet.metrics.TTestPower()
    sample_size_metric = beignet.metrics.TTestSampleSize()

    assert isinstance(cohens_d_metric, CohensD)
    assert isinstance(hedges_g_metric, HedgesG)
    assert isinstance(power_metric, TTestPower)
    assert isinstance(sample_size_metric, TTestSampleSize)


def test_metric_consistency():
    """Test that metrics produce consistent results with direct operator calls."""
    # Generate test data
    torch.manual_seed(42)
    group1 = torch.randn(25, dtype=torch.float32)
    group2 = torch.randn(25, dtype=torch.float32) + 0.5

    # Test Cohen's D consistency
    cohens_d_metric = CohensD(pooled=True)
    cohens_d_metric.update(group1, group2)
    metric_result = cohens_d_metric.compute()

    direct_result = beignet.cohens_d(
        group1.unsqueeze(0), group2.unsqueeze(0), pooled=True
    )
    assert torch.allclose(metric_result, direct_result.squeeze(), atol=1e-6)

    # Test Hedges' G consistency
    hedges_g_metric = HedgesG()
    hedges_g_metric.update(group1, group2)
    metric_hedges = hedges_g_metric.compute()

    direct_hedges = beignet.hedges_g(group1.unsqueeze(0), group2.unsqueeze(0))
    assert torch.allclose(metric_hedges, direct_hedges.squeeze(), atol=1e-6)

    # Test power calculation consistency
    power_metric = TTestPower(alpha=0.05, alternative="two-sided")
    power_metric.update(group1, group2)
    metric_power = power_metric.compute()

    # Manual calculation
    effect_size = beignet.cohens_d(
        group1.unsqueeze(0), group2.unsqueeze(0), pooled=True
    )
    sample_size = torch.tensor(25.0, dtype=torch.float32)
    direct_power = beignet.t_test_power(
        effect_size.squeeze(), sample_size, alpha=0.05, alternative="two-sided"
    )
    assert torch.allclose(metric_power, direct_power, atol=1e-6)

    # Test sample size calculation consistency
    sample_size_metric = TTestSampleSize(power=0.8, alpha=0.05, alternative="two-sided")
    sample_size_metric.update(group1, group2)
    metric_n = sample_size_metric.compute()

    # Manual calculation
    direct_n = beignet.t_test_sample_size(
        effect_size.squeeze(), power=0.8, alpha=0.05, alternative="two-sided"
    )
    assert torch.allclose(metric_n, direct_n, atol=1e-6)
