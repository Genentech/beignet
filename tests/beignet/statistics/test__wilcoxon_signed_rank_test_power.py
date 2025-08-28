import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_wilcoxon_signed_rank_test_power(batch_size, dtype):
    """Test Wilcoxon signed-rank test power calculation."""

    # Generate test parameters - prob_positive should be > 0.5 for positive effect
    prob_positive = (
        torch.tensor([0.55, 0.65, 0.75], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([15, 25, 40], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.wilcoxon_signed_rank_test_power(
        prob_positive,
        sample_sizes,
    )
    assert result.shape == prob_positive.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(prob_positive)
    result_out = beignet.statistics.wilcoxon_signed_rank_test_power(
        prob_positive,
        sample_sizes,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with higher prob_positive (stronger effect)
    small_prob = beignet.statistics.wilcoxon_signed_rank_test_power(
        torch.tensor(0.55, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
    )
    large_prob = beignet.statistics.wilcoxon_signed_rank_test_power(
        torch.tensor(0.75, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
    )
    assert large_prob > small_prob

    # Test that power increases with sample size
    small_n = beignet.statistics.wilcoxon_signed_rank_test_power(
        torch.tensor(0.65, dtype=dtype),
        torch.tensor(15.0, dtype=dtype),
    )
    large_n = beignet.statistics.wilcoxon_signed_rank_test_power(
        torch.tensor(0.65, dtype=dtype),
        torch.tensor(40.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test different alpha values
    power_05 = beignet.statistics.wilcoxon_signed_rank_test_power(
        torch.tensor(0.65, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        alpha=0.05,
    )
    power_01 = beignet.statistics.wilcoxon_signed_rank_test_power(
        torch.tensor(0.65, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        alpha=0.01,
    )
    assert power_05 > power_01

    # Test gradient computation
    prob_grad = torch.tensor([0.65], dtype=dtype, requires_grad=True)
    sample_grad = torch.tensor([25.0], dtype=dtype, requires_grad=True)
    result_grad = beignet.statistics.wilcoxon_signed_rank_test_power(
        prob_grad,
        sample_grad,
    )

    loss = result_grad.sum()
    loss.backward()

    assert prob_grad.grad is not None
    assert sample_grad.grad is not None
    assert torch.all(torch.isfinite(prob_grad.grad))
    assert torch.all(torch.isfinite(sample_grad.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        beignet.statistics.wilcoxon_signed_rank_test_power,
        fullgraph=True,
    )
    result_compiled = compiled_func(prob_positive[:1], sample_sizes[:1])
    result_normal = beignet.statistics.wilcoxon_signed_rank_test_power(
        prob_positive[:1],
        sample_sizes[:1],
    )
    assert torch.allclose(result_compiled, result_normal, atol=1e-6)
