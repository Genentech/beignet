import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    n_elements=hypothesis.strategies.integers(min_value=5, max_value=20),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_eta_squared(batch_size, n_elements, dtype):
    """Test eta squared effect size calculation."""
    # Generate test data - sum of squares between groups and total sum of squares
    # In ANOVA context: eta² = SS_between / SS_total
    # Both tensors must have the same shape for element-wise operations
    ss_between = torch.rand(batch_size, n_elements, dtype=dtype) * 100.0  # Non-negative
    ss_total = (
        torch.rand(batch_size, n_elements, dtype=dtype) * 100.0 + 50.0
    )  # Ensure > ss_between generally

    # Test basic functionality
    result = beignet.statistics.eta_squared(ss_between, ss_total)
    assert result.shape == (batch_size, n_elements)  # Preserves input shape
    assert result.dtype == torch.promote_types(ss_between.dtype, ss_total.dtype)

    # Test that result is bounded between 0 and 1
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with zero between-group variance (no effect)
    zeros = torch.zeros(batch_size, n_elements, dtype=dtype)
    total_variance = (
        torch.rand(batch_size, n_elements, dtype=dtype) + 1.0
    )  # Positive total
    result_zero = beignet.statistics.eta_squared(zeros, total_variance)
    assert torch.allclose(result_zero, torch.zeros_like(result_zero), atol=1e-6)

    # Test with equal between-group and total variance (perfect effect)
    equal_variance = torch.rand(batch_size, n_elements, dtype=dtype) + 1.0
    result_perfect = beignet.statistics.eta_squared(equal_variance, equal_variance)
    assert torch.allclose(result_perfect, torch.ones_like(result_perfect), atol=1e-6)

    # Test with out parameter
    out = torch.empty_like(result)
    result_out = beignet.statistics.eta_squared(ss_between, ss_total, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test gradient computation
    ss_between_grad = ss_between.clone().requires_grad_(True)
    ss_total_grad = ss_total.clone().requires_grad_(True)
    result_grad = beignet.statistics.eta_squared(ss_between_grad, ss_total_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert ss_between_grad.grad is not None
    assert ss_total_grad.grad is not None
    assert ss_between_grad.grad.shape == ss_between.shape
    assert ss_total_grad.grad.shape == ss_total.shape

    # Test torch.compile compatibility
    compiled_eta_squared = torch.compile(beignet.statistics.eta_squared, fullgraph=True)
    result_compiled = compiled_eta_squared(ss_between, ss_total)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test clamping behavior - negative values should be clamped to 0
    negative_between = torch.tensor([[-5.0, -2.0]], dtype=dtype)
    positive_total = torch.tensor([[10.0, 8.0]], dtype=dtype)
    result_negative = beignet.statistics.eta_squared(negative_between, positive_total)
    assert torch.all(result_negative >= 0.0)
    expected_negative = torch.zeros_like(result_negative)  # Negative input clamped to 0
    assert torch.allclose(result_negative, expected_negative, atol=1e-6)

    # Test clamping behavior - between > total should be clamped
    large_between = torch.tensor([[15.0, 20.0]], dtype=dtype)
    small_total = torch.tensor([[10.0, 12.0]], dtype=dtype)
    result_clamped = beignet.statistics.eta_squared(large_between, small_total)
    assert torch.all(result_clamped <= 1.0)
    expected_clamped = torch.ones_like(result_clamped)  # Should be clamped to 1.0
    assert torch.allclose(result_clamped, expected_clamped, atol=1e-6)

    # Test with very small total values (should use eps)
    small_between = torch.tensor([[1.0]], dtype=dtype)
    tiny_total = torch.tensor([[1e-10]], dtype=dtype)  # Much smaller than eps
    result_tiny = beignet.statistics.eta_squared(small_between, tiny_total)
    # The tiny total should be clamped to eps, so result should be bounded
    assert torch.all(result_tiny >= 0.0)
    assert torch.all(result_tiny <= 1.0)

    # Test known mathematical examples
    # Case 1: Between SS = 25, Total SS = 100 -> eta² = 0.25
    known_between = torch.tensor([[25.0]], dtype=dtype)
    known_total = torch.tensor([[100.0]], dtype=dtype)
    result_known = beignet.statistics.eta_squared(known_between, known_total)
    expected_known = torch.tensor([[0.25]], dtype=dtype)
    assert torch.allclose(result_known, expected_known, atol=1e-6)

    # Case 2: Between SS = 0, Total SS = 50 -> eta² = 0.0
    zero_between = torch.tensor([[0.0]], dtype=dtype)
    some_total = torch.tensor([[50.0]], dtype=dtype)
    result_zero_known = beignet.statistics.eta_squared(zero_between, some_total)
    expected_zero = torch.tensor([[0.0]], dtype=dtype)
    assert torch.allclose(result_zero_known, expected_zero, atol=1e-6)

    # Test single values (1D tensors preserve shape)
    single_between = torch.tensor([4.0], dtype=dtype)
    single_total = torch.tensor([16.0], dtype=dtype)
    result_single = beignet.statistics.eta_squared(single_between, single_total)
    assert result_single.shape == (1,)  # Preserves 1D shape
    expected_single = torch.tensor([0.25], dtype=dtype)  # 4/16 = 0.25
    assert torch.allclose(result_single, expected_single, atol=1e-6)

    # Test type promotion
    float32_between = torch.tensor([[2.0]], dtype=torch.float32)
    float64_total = torch.tensor([[8.0]], dtype=torch.float64)
    result_promoted = beignet.statistics.eta_squared(float32_between, float64_total)
    assert result_promoted.dtype == torch.float64

    # Test 1D input handling (preserves 1D shape)
    between_1d = torch.tensor([6.0], dtype=dtype)
    total_1d = torch.tensor([24.0], dtype=dtype)
    result_1d = beignet.statistics.eta_squared(between_1d, total_1d)
    assert result_1d.shape == (1,)  # Preserves 1D shape
    expected_1d = torch.tensor([0.25], dtype=dtype)  # 6/24 = 0.25
    assert torch.allclose(result_1d, expected_1d, atol=1e-6)

    # Test edge case: both values very small but positive (above eps)
    tiny_between = torch.tensor([[1e-6]], dtype=dtype)  # Above eps for both float32/64
    tiny_total = torch.tensor([[1e-5]], dtype=dtype)  # Above eps for both float32/64
    result_tiny_both = beignet.statistics.eta_squared(tiny_between, tiny_total)
    expected_tiny = torch.tensor([[0.1]], dtype=dtype)  # 1e-6 / 1e-5 = 0.1
    assert torch.allclose(result_tiny_both, expected_tiny, atol=1e-5)

    # Test batch processing with mixed scenarios
    mixed_between = torch.tensor(
        [[0.0], [25.0], [100.0]],
        dtype=dtype,
    )  # No effect, medium, large
    mixed_total = torch.tensor([[10.0], [100.0], [100.0]], dtype=dtype)
    result_mixed = beignet.statistics.eta_squared(mixed_between, mixed_total)
    expected_mixed = torch.tensor([[0.0], [0.25], [1.0]], dtype=dtype)
    assert torch.allclose(result_mixed, expected_mixed, atol=1e-6)

    # Test monotonicity: increasing between-group variance should increase eta²
    # when total variance is held constant
    fixed_total = torch.tensor([[100.0]], dtype=dtype)
    small_between_mono = torch.tensor([[10.0]], dtype=dtype)
    large_between_mono = torch.tensor([[50.0]], dtype=dtype)

    result_small_mono = beignet.statistics.eta_squared(small_between_mono, fixed_total)
    result_large_mono = beignet.statistics.eta_squared(large_between_mono, fixed_total)

    assert torch.all(result_large_mono > result_small_mono)

    # Test that eta² = 0 when input is zero regardless of other value
    zero_effect = torch.zeros(2, 3, dtype=dtype)
    various_totals = torch.rand(2, 3, dtype=dtype) + 1.0
    result_always_zero = beignet.statistics.eta_squared(zero_effect, various_totals)
    assert torch.allclose(result_always_zero, torch.zeros_like(result_always_zero))

    # Test numerical stability with extreme values
    large_between = torch.tensor([[1e6]], dtype=dtype)
    large_total = torch.tensor([[1e7]], dtype=dtype)
    result_large = beignet.statistics.eta_squared(large_between, large_total)
    expected_large = torch.tensor([[0.1]], dtype=dtype)  # 1e6 / 1e7 = 0.1
    assert torch.allclose(result_large, expected_large, rtol=1e-5)
