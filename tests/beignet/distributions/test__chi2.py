import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.distributions import Chi2


def test_chi2_basic():
    """Test basic Chi2 functionality."""
    df = torch.tensor(5.0)
    dist = Chi2(df)

    # Test icdf with known values
    quantile = dist.icdf(torch.tensor(0.95))

    # Should be close to 11.07 for df=5, alpha=0.05
    assert torch.allclose(quantile, torch.tensor(11.07), atol=0.1)


@given(
    df=st.floats(min_value=1.0, max_value=50.0),
    prob=st.floats(min_value=0.01, max_value=0.99),
)
def test_chi2_properties(df, prob):
    """Test Chi2 distribution properties with hypothesis."""
    df_tensor = torch.tensor(df, dtype=torch.float32)
    prob_tensor = torch.tensor(prob, dtype=torch.float32)

    dist = Chi2(df_tensor)
    quantile = dist.icdf(prob_tensor)

    # Quantile should be finite and non-negative
    assert torch.isfinite(quantile)
    assert quantile >= 0.0

    # For higher probabilities, quantile should be higher
    if prob > 0.5:
        lower_quantile = dist.icdf(torch.tensor(0.5))
        assert quantile > lower_quantile


def test_chi2_batch():
    """Test Chi2 with batch operations."""
    df = torch.tensor([1.0, 5.0, 10.0])
    probs = torch.tensor([0.05, 0.5, 0.95])

    dist = Chi2(df)
    quantiles = dist.icdf(probs)

    # Should have same shape
    assert quantiles.shape == df.shape

    # All quantiles should be non-negative
    assert torch.all(quantiles >= 0.0)


def test_chi2_torch_compile():
    """Test torch.compile compatibility."""
    df = torch.tensor(5.0)
    dist = Chi2(df)

    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    prob = torch.tensor(0.9)
    result = dist.icdf(prob)
    result_compiled = compiled_icdf(prob)

    assert torch.allclose(result, result_compiled, atol=1e-6)


def test_chi2_large_df():
    """Test Chi2 with large degrees of freedom."""
    df = torch.tensor(100.0)
    dist = Chi2(df)

    # For large df, chi2 approaches normal distribution
    quantile_50 = dist.icdf(torch.tensor(0.5))

    # Should be close to df for p=0.5
    assert torch.allclose(quantile_50, df, rtol=0.1)


def test_chi2_small_df():
    """Test Chi2 with small degrees of freedom."""
    df = torch.tensor(1.0)
    dist = Chi2(df)

    # Test various quantiles
    probs = torch.tensor([0.1, 0.5, 0.9])
    quantiles = dist.icdf(probs)

    # All should be finite and non-negative
    assert torch.all(torch.isfinite(quantiles))
    assert torch.all(quantiles >= 0.0)

    # Should be monotonic
    assert quantiles[0] < quantiles[1] < quantiles[2]


def test_chi2_edge_cases():
    """Test edge cases."""
    df = torch.tensor(2.0)
    dist = Chi2(df)

    # Should handle edge probabilities
    eps = torch.finfo(torch.float32).eps

    low_quantile = dist.icdf(torch.tensor(eps))
    high_quantile = dist.icdf(torch.tensor(1.0 - eps))

    assert torch.isfinite(low_quantile)
    assert torch.isfinite(high_quantile)
    assert low_quantile >= 0.0
    assert high_quantile > low_quantile
