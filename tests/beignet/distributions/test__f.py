import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.distributions import F


def test_f_basic():
    """Test basic F-distribution functionality."""
    df1 = torch.tensor(3.0)
    df2 = torch.tensor(20.0)
    dist = F(df1, df2)

    # Test icdf with known values (approximate)
    quantile = dist.icdf(torch.tensor(0.95))

    # Should be around 3.10 for F(3,20) at p=0.95 (allowing some approximation error)
    assert 2.5 < quantile.item() < 3.5


@given(
    df1=st.floats(min_value=1.0, max_value=50.0),
    df2=st.floats(min_value=2.0, max_value=50.0),
    prob=st.floats(min_value=0.01, max_value=0.99),
)
def test_f_properties(df1, df2, prob):
    """Test F-distribution properties with hypothesis."""
    df1_tensor = torch.tensor(df1, dtype=torch.float32)
    df2_tensor = torch.tensor(df2, dtype=torch.float32)
    prob_tensor = torch.tensor(prob, dtype=torch.float32)

    dist = F(df1_tensor, df2_tensor)
    quantile = dist.icdf(prob_tensor)

    # Quantile should be finite and positive
    assert torch.isfinite(quantile)
    assert quantile > 0.0

    # For higher probabilities, quantile should generally be higher
    if prob > 0.5:
        lower_quantile = dist.icdf(torch.tensor(0.5))
        assert quantile > lower_quantile


def test_f_batch():
    """Test F-distribution with batch operations."""
    df1 = torch.tensor([1.0, 5.0, 10.0])
    df2 = torch.tensor([10.0, 10.0, 10.0])
    probs = torch.tensor([0.05, 0.5, 0.95])

    dist = F(df1, df2)
    quantiles = dist.icdf(probs)

    # Should have same shape
    assert quantiles.shape == df1.shape

    # All quantiles should be positive
    assert torch.all(quantiles > 0.0)

    # Should be roughly monotonic for fixed df
    assert quantiles[0] < quantiles[1] < quantiles[2]


def test_f_torch_compile():
    """Test torch.compile compatibility."""
    df1 = torch.tensor(5.0)
    df2 = torch.tensor(10.0)
    dist = F(df1, df2)

    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    prob = torch.tensor(0.9)
    result = dist.icdf(prob)
    result_compiled = compiled_icdf(prob)

    assert torch.allclose(result, result_compiled, atol=1e-6)


def test_f_edge_cases():
    """Test edge cases."""
    # Small degrees of freedom
    small_dist = F(torch.tensor(1.0), torch.tensor(2.0))

    # Should handle edge probabilities
    low_quantile = small_dist.icdf(torch.tensor(0.01))
    high_quantile = small_dist.icdf(torch.tensor(0.99))

    assert torch.isfinite(low_quantile)
    assert torch.isfinite(high_quantile)
    assert low_quantile > 0.0
    assert high_quantile >= low_quantile


def test_f_large_df():
    """Test F-distribution with large degrees of freedom."""
    # For large df1 and df2, F should approach 1.0 for p=0.5
    large_dist = F(torch.tensor(100.0), torch.tensor(100.0))

    median_quantile = large_dist.icdf(torch.tensor(0.5))

    # Should be close to 1.0 for large df
    assert 0.8 < median_quantile.item() < 1.2


def test_f_asymmetry():
    """Test that F-distribution is properly asymmetric."""
    dist = F(torch.tensor(5.0), torch.tensor(10.0))

    # Test various quantiles
    probs = torch.tensor([0.1, 0.5, 0.9])
    quantiles = dist.icdf(probs)

    # F-distribution should be right-skewed
    # So distance from median to 90th percentile should be > distance from 10th to median
    median = quantiles[1]
    lower_distance = median - quantiles[0]
    upper_distance = quantiles[2] - median

    assert upper_distance > lower_distance


def test_f_monotonicity():
    """Test that F icdf is monotonic."""
    dist = F(torch.tensor(3.0), torch.tensor(15.0))

    probs = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    quantiles = dist.icdf(probs)

    # Should be strictly increasing
    for i in range(len(quantiles) - 1):
        assert quantiles[i] < quantiles[i + 1]


def test_f_different_df_combinations():
    """Test F-distribution with various degrees of freedom combinations."""
    test_cases = [
        (1.0, 1.0),  # Both small
        (1.0, 100.0),  # Small df1, large df2
        (100.0, 1.0),  # Large df1, small df2
        (50.0, 50.0),  # Both large
    ]

    prob = torch.tensor(0.95)

    for df1, df2 in test_cases:
        dist = F(torch.tensor(df1), torch.tensor(df2))
        quantile = dist.icdf(prob)

        # Should always be finite and positive
        assert torch.isfinite(quantile)
        assert quantile > 0.0
