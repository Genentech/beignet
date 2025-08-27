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
def test_partial_eta_squared(batch_size, n_elements, dtype):
    """Test partial eta squared effect size calculation."""
    # Generate test data - sum of squares for effect and error
    # In ANOVA context: partial η² = SS_effect / (SS_effect + SS_error)
    # Both tensors must have compatible shapes for element-wise operations
    ss_effect = torch.rand(batch_size, n_elements, dtype=dtype) * 100.0  # Non-negative
    ss_error = (
        torch.rand(batch_size, n_elements, dtype=dtype) * 100.0 + 10.0
    )  # Positive

    # Test basic functionality
    result = beignet.statistics.partial_eta_squared(ss_effect, ss_error)
    assert result.shape == (batch_size, n_elements)  # Preserves input shape
    assert result.dtype == torch.promote_types(ss_effect.dtype, ss_error.dtype)

    # Test that result is bounded between 0 and 1
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with zero effect (no variance explained)
    zeros = torch.zeros(batch_size, n_elements, dtype=dtype)
    error_variance = torch.rand(batch_size, n_elements, dtype=dtype) + 1.0  # Positive
    result_zero = beignet.statistics.partial_eta_squared(zeros, error_variance)
    assert torch.allclose(result_zero, torch.zeros_like(result_zero), atol=1e-6)

    # Test with zero error (perfect effect, but clamped by eps)
    effect_variance = torch.rand(batch_size, n_elements, dtype=dtype) + 1.0
    tiny_error = torch.zeros(batch_size, n_elements, dtype=dtype)
    result_perfect = beignet.statistics.partial_eta_squared(effect_variance, tiny_error)
    # Should approach 1.0 but be bounded due to eps clamping
    assert torch.all(result_perfect > 0.99)
    assert torch.all(result_perfect <= 1.0)

    # Test with out parameter
    out = torch.empty_like(result)
    result_out = beignet.statistics.partial_eta_squared(ss_effect, ss_error, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test gradient computation
    ss_effect_grad = ss_effect.clone().requires_grad_(True)
    ss_error_grad = ss_error.clone().requires_grad_(True)
    result_grad = beignet.statistics.partial_eta_squared(ss_effect_grad, ss_error_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert ss_effect_grad.grad is not None
    assert ss_error_grad.grad is not None
    assert ss_effect_grad.grad.shape == ss_effect.shape
    assert ss_error_grad.grad.shape == ss_error.shape

    # Test torch.compile compatibility
    compiled_partial_eta_squared = torch.compile(
        beignet.statistics.partial_eta_squared,
        fullgraph=True,
    )
    result_compiled = compiled_partial_eta_squared(ss_effect, ss_error)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test mathematical properties - partial eta squared formula verification
    # Partial η² = SS_effect / (SS_effect + SS_error)
    expected_partial_eta = ss_effect / (
        ss_effect + torch.clamp(ss_error, min=torch.finfo(dtype).eps)
    )
    expected_partial_eta = torch.clamp(expected_partial_eta, 0.0, 1.0)
    assert torch.allclose(result, expected_partial_eta, atol=1e-6)

    # Test with known values for deterministic verification
    # Case 1: Effect SS = 25, Error SS = 75 -> partial η² = 25/(25+75) = 0.25
    known_effect = torch.tensor([[25.0]], dtype=dtype)
    known_error = torch.tensor([[75.0]], dtype=dtype)
    result_known = beignet.statistics.partial_eta_squared(known_effect, known_error)
    expected_known = torch.tensor([[0.25]], dtype=dtype)
    assert torch.allclose(result_known, expected_known, atol=1e-6)

    # Case 2: Effect SS = 80, Error SS = 20 -> partial η² = 80/(80+20) = 0.8
    large_effect = torch.tensor([[80.0]], dtype=dtype)
    small_error = torch.tensor([[20.0]], dtype=dtype)
    result_large = beignet.statistics.partial_eta_squared(large_effect, small_error)
    expected_large = torch.tensor([[0.8]], dtype=dtype)
    assert torch.allclose(result_large, expected_large, atol=1e-6)

    # Test clamping behavior - negative values should be clamped to 0
    negative_effect = torch.tensor([[-10.0]], dtype=dtype)
    positive_error = torch.tensor([[50.0]], dtype=dtype)
    result_negative = beignet.statistics.partial_eta_squared(
        negative_effect,
        positive_error,
    )
    expected_negative = torch.tensor([[0.0]], dtype=dtype)  # Negative clamped to 0
    assert torch.allclose(result_negative, expected_negative, atol=1e-6)

    # Test very small error values (eps clamping)
    moderate_effect = torch.tensor([[30.0]], dtype=dtype)
    tiny_error_val = torch.tensor([[1e-12]], dtype=dtype)  # Much smaller than eps
    result_tiny_error = beignet.statistics.partial_eta_squared(
        moderate_effect,
        tiny_error_val,
    )
    # Error should be clamped to eps, so result should be bounded
    assert torch.all(result_tiny_error >= 0.0)
    assert torch.all(result_tiny_error <= 1.0)

    # Test type promotion
    float32_effect = torch.tensor([[20.0]], dtype=torch.float32)
    float64_error = torch.tensor([[80.0]], dtype=torch.float64)
    result_promoted = beignet.statistics.partial_eta_squared(
        float32_effect,
        float64_error,
    )
    assert result_promoted.dtype == torch.float64

    # Test 1D input handling (preserves shape)
    effect_1d = torch.tensor([40.0], dtype=dtype)
    error_1d = torch.tensor([60.0], dtype=dtype)
    result_1d = beignet.statistics.partial_eta_squared(effect_1d, error_1d)
    assert result_1d.shape == (1,)  # Preserves 1D shape
    expected_1d = torch.tensor([0.4], dtype=dtype)  # 40/(40+60) = 0.4
    assert torch.allclose(result_1d, expected_1d, atol=1e-6)

    # Test difference from eta squared
    # Partial η² uses only effect and error SS, while η² uses effect and total SS
    # For comparison, create scenarios where partial η² and η² would differ
    effect_ss = torch.tensor([[30.0]], dtype=dtype)
    error_ss = torch.tensor([[70.0]], dtype=dtype)
    total_ss = torch.tensor([[120.0]], dtype=dtype)  # > effect + error (other sources)

    partial_eta_result = beignet.statistics.partial_eta_squared(effect_ss, error_ss)
    eta_result = beignet.statistics.eta_squared(effect_ss, total_ss)

    # Partial η² = 30/(30+70) = 0.3, η² = 30/120 = 0.25
    expected_partial = 30.0 / (30.0 + 70.0)  # 0.3
    expected_eta = 30.0 / 120.0  # 0.25

    assert torch.allclose(
        partial_eta_result,
        torch.tensor([[expected_partial]], dtype=dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        eta_result,
        torch.tensor([[expected_eta]], dtype=dtype),
        atol=1e-6,
    )
    assert partial_eta_result > eta_result  # Partial should be larger

    # Test monotonicity: increasing effect SS should increase partial η²
    fixed_error = torch.tensor([[50.0]], dtype=dtype)
    small_effect_mono = torch.tensor([[10.0]], dtype=dtype)
    large_effect_mono = torch.tensor([[40.0]], dtype=dtype)

    result_small_mono = beignet.statistics.partial_eta_squared(
        small_effect_mono,
        fixed_error,
    )
    result_large_mono = beignet.statistics.partial_eta_squared(
        large_effect_mono,
        fixed_error,
    )

    assert result_large_mono > result_small_mono

    # Test monotonicity: increasing error SS should decrease partial η²
    fixed_effect = torch.tensor([[30.0]], dtype=dtype)
    small_error_mono = torch.tensor([[10.0]], dtype=dtype)
    large_error_mono = torch.tensor([[90.0]], dtype=dtype)

    result_small_error = beignet.statistics.partial_eta_squared(
        fixed_effect,
        small_error_mono,
    )
    result_large_error = beignet.statistics.partial_eta_squared(
        fixed_effect,
        large_error_mono,
    )

    assert result_small_error > result_large_error

    # Test batch processing with mixed scenarios
    mixed_effect = torch.tensor(
        [[0.0], [25.0], [80.0]],
        dtype=dtype,
    )  # No effect, medium, large
    mixed_error = torch.tensor([[50.0], [75.0], [20.0]], dtype=dtype)
    result_mixed = beignet.statistics.partial_eta_squared(mixed_effect, mixed_error)

    expected_mixed = torch.tensor(
        [
            [0.0],  # 0/(0+50) = 0
            [0.25],  # 25/(25+75) = 0.25
            [0.8],  # 80/(80+20) = 0.8
        ],
        dtype=dtype,
    )
    assert torch.allclose(result_mixed, expected_mixed, atol=1e-6)

    # Test numerical stability with extreme values
    large_effect = torch.tensor([[1e6]], dtype=dtype)
    large_error = torch.tensor([[9e6]], dtype=dtype)
    result_extreme = beignet.statistics.partial_eta_squared(large_effect, large_error)
    expected_extreme = torch.tensor([[0.1]], dtype=dtype)  # 1e6/(1e6+9e6) = 0.1
    assert torch.allclose(result_extreme, expected_extreme, rtol=1e-5)

    # Test interpretation: partial η² measures proportion of relevant variance
    # It excludes other sources of variance not related to the effect being tested
    effect_only = torch.tensor([[45.0]], dtype=dtype)
    error_only = torch.tensor([[55.0]], dtype=dtype)

    partial_result = beignet.statistics.partial_eta_squared(effect_only, error_only)
    expected_proportion = 45.0 / (45.0 + 55.0)  # 0.45

    assert torch.allclose(
        partial_result,
        torch.tensor([[expected_proportion]], dtype=dtype),
        atol=1e-6,
    )

    # Test edge case: equal effect and error sums of squares
    equal_ss = torch.tensor([[50.0]], dtype=dtype)
    result_equal = beignet.statistics.partial_eta_squared(equal_ss, equal_ss)
    expected_equal = torch.tensor([[0.5]], dtype=dtype)  # 50/(50+50) = 0.5
    assert torch.allclose(result_equal, expected_equal, atol=1e-6)

    # Test batch consistency with different effect-to-error ratios
    ratio_effects = torch.tensor([[10.0], [20.0], [40.0]], dtype=dtype)
    ratio_errors = torch.tensor(
        [[90.0], [80.0], [60.0]],
        dtype=dtype,
    )  # Decreasing error

    ratio_results = beignet.statistics.partial_eta_squared(ratio_effects, ratio_errors)

    # Expected: [0.1, 0.2, 0.4] - increasing as error decreases
    expected_ratios = torch.tensor([[0.1], [0.2], [0.4]], dtype=dtype)
    assert torch.allclose(ratio_results, expected_ratios, atol=1e-6)

    # Verify increasing trend
    assert ratio_results[1] > ratio_results[0]
    assert ratio_results[2] > ratio_results[1]

    # Test limiting behavior
    # As effect approaches infinity relative to error, partial η² approaches 1
    huge_effect = torch.tensor([[1e8]], dtype=dtype)
    small_error_limit = torch.tensor([[1.0]], dtype=dtype)
    result_limit = beignet.statistics.partial_eta_squared(
        huge_effect,
        small_error_limit,
    )

    # Should be very close to 1.0
    assert result_limit > 0.99999
    assert result_limit <= 1.0

    # Test that partial η² = 0 when effect is zero regardless of error value
    zero_effect_test = torch.zeros(2, 3, dtype=dtype)
    various_errors = torch.rand(2, 3, dtype=dtype) + 1.0
    result_always_zero = beignet.statistics.partial_eta_squared(
        zero_effect_test,
        various_errors,
    )
    assert torch.allclose(result_always_zero, torch.zeros_like(result_always_zero))
