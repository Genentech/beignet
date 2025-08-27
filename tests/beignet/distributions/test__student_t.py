import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.distributions import StudentT


def test_student_t_basic():
    """Test basic StudentT functionality."""
    df = torch.tensor(10.0)
    dist = StudentT(df)

    # Test icdf with known values
    quantile = dist.icdf(torch.tensor(0.975))

    # Should be close to 2.228 for df=10, alpha=0.025
    assert torch.allclose(quantile, torch.tensor(2.228), atol=0.01)

    # Test symmetry
    lower_quantile = dist.icdf(torch.tensor(0.025))
    assert torch.allclose(quantile, -lower_quantile, atol=1e-4)


@given(
    df=st.floats(min_value=1.0, max_value=100.0),
    prob=st.floats(min_value=0.01, max_value=0.99),
)
def test_student_t_properties(df, prob):
    """Test StudentT distribution properties with hypothesis."""
    df_tensor = torch.tensor(df, dtype=torch.float32)
    prob_tensor = torch.tensor(prob, dtype=torch.float32)

    dist = StudentT(df_tensor)
    quantile = dist.icdf(prob_tensor)

    # Quantile should be finite
    assert torch.isfinite(quantile)

    # Test with location and scale
    loc = torch.tensor(1.0)
    scale = torch.tensor(2.0)
    dist_scaled = StudentT(df_tensor, loc=loc, scale=scale)
    quantile_scaled = dist_scaled.icdf(prob_tensor)

    # Should satisfy: quantile_scaled = loc + scale * quantile_standard
    expected = loc + scale * quantile
    assert torch.allclose(quantile_scaled, expected, atol=1e-5)


def test_student_t_batch():
    """Test StudentT with batch operations."""
    df = torch.tensor([5.0, 10.0, 20.0])
    probs = torch.tensor([0.025, 0.5, 0.975])

    dist = StudentT(df)
    quantiles = dist.icdf(probs)

    # Should have same shape
    assert quantiles.shape == df.shape

    # Middle quantile should be near 0 (symmetric distribution)
    assert torch.allclose(quantiles[1], torch.tensor(0.0), atol=1e-3)


def test_student_t_torch_compile():
    """Test torch.compile compatibility."""
    df = torch.tensor(10.0)
    dist = StudentT(df)

    compiled_icdf = torch.compile(dist.icdf, fullgraph=True)

    prob = torch.tensor(0.9)
    result = dist.icdf(prob)
    result_compiled = compiled_icdf(prob)

    assert torch.allclose(result, result_compiled, atol=1e-6)


def test_student_t_edge_cases():
    """Test edge cases."""
    df = torch.tensor(1.0)  # t(1) is Cauchy distribution
    dist = StudentT(df)

    # Should handle edge probabilities
    eps = torch.finfo(torch.float32).eps

    low_quantile = dist.icdf(torch.tensor(eps))
    high_quantile = dist.icdf(torch.tensor(1.0 - eps))

    assert torch.isfinite(low_quantile)
    assert torch.isfinite(high_quantile)
    assert low_quantile < high_quantile
