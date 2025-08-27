import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    n1=hypothesis.strategies.integers(min_value=5, max_value=20),
    n2=hypothesis.strategies.integers(min_value=5, max_value=20),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_glass_delta(batch_size, n1, n2, dtype):
    """Test Glass' delta effect size calculation."""
    # Generate test data
    group1 = torch.randn(batch_size, n1, dtype=dtype)
    group2 = torch.randn(batch_size, n2, dtype=dtype)

    # Test basic functionality
    result = beignet.statistics.glass_delta(group1, group2)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype

    # Test with identical groups (should be close to zero)
    identical_group = torch.randn(batch_size, n1, dtype=dtype)
    result_identical = beignet.statistics.glass_delta(identical_group, identical_group)
    assert torch.allclose(
        result_identical,
        torch.zeros_like(result_identical),
        atol=1e-6,
    )

    # Test asymmetry property: Glass' delta(A, B) ≠ -Glass' delta(B, A)
    # Unlike Cohen's d, Glass' delta is not symmetric because it uses only group2's SD
    result_forward = beignet.statistics.glass_delta(group1, group2)
    result_backward = beignet.statistics.glass_delta(group2, group1)
    # Should not be equal due to different denominators (different variances)
    assert not torch.allclose(result_forward, -result_backward, atol=1e-5)

    # Test with out parameter
    out = torch.empty(batch_size, dtype=dtype)
    result_out = beignet.statistics.glass_delta(group1, group2, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)
    result_grad = beignet.statistics.glass_delta(group1_grad, group2_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test torch.compile compatibility
    compiled_glass_delta = torch.compile(beignet.statistics.glass_delta, fullgraph=True)
    result_compiled = compiled_glass_delta(group1, group2)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test mathematical properties - Glass' delta formula verification
    # Glass' delta = (mean1 - mean2) / std2
    mean1 = torch.mean(group1, dim=-1)
    mean2 = torch.mean(group2, dim=-1)
    std2 = torch.sqrt(torch.var(group2, dim=-1, correction=1))

    expected_glass_delta = (mean1 - mean2) / torch.clamp(
        std2,
        min=torch.finfo(dtype).eps,
    )
    assert torch.allclose(result, expected_glass_delta, atol=1e-6)

    # Test with known values for deterministic verification
    test_group1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=dtype)  # mean = 3.0
    test_group2 = torch.tensor([[0.0, 2.0, 4.0]], dtype=dtype)  # mean = 2.0, std = 2.0

    # Glass' delta = (3.0 - 2.0) / 2.0 = 0.5
    result_known = beignet.statistics.glass_delta(test_group1, test_group2)
    assert torch.allclose(result_known, torch.tensor([0.5], dtype=dtype), atol=1e-6)

    # Test with constant variance in control group (group2)
    # Glass' delta should scale linearly with mean difference when group2 variance is constant
    constant_group2 = torch.tensor([[1.0, 3.0, 5.0]], dtype=dtype)  # std = 2.0

    small_diff_group1 = torch.tensor([[2.0, 2.0, 2.0]], dtype=dtype)  # mean = 2.0
    large_diff_group1 = torch.tensor([[5.0, 5.0, 5.0]], dtype=dtype)  # mean = 5.0

    result_small = beignet.statistics.glass_delta(small_diff_group1, constant_group2)
    result_large = beignet.statistics.glass_delta(large_diff_group1, constant_group2)

    # mean_diff_small = 2.0 - 3.0 = -1.0, Glass' delta = -1.0/2.0 = -0.5
    # mean_diff_large = 5.0 - 3.0 = 2.0, Glass' delta = 2.0/2.0 = 1.0
    assert torch.allclose(result_small, torch.tensor([-0.5], dtype=dtype), atol=1e-6)
    assert torch.allclose(result_large, torch.tensor([1.0], dtype=dtype), atol=1e-6)

    # Test with zero variance in control group (edge case)
    zero_var_group2 = torch.ones(1, 5, dtype=dtype) * 3.0  # All same value
    test_group1_zero = torch.tensor([[1.0, 2.0, 4.0, 5.0, 6.0]], dtype=dtype)

    result_zero_var = beignet.statistics.glass_delta(test_group1_zero, zero_var_group2)
    # Should handle gracefully with eps clamping (likely very large value)
    assert torch.all(torch.isfinite(result_zero_var) | torch.isinf(result_zero_var))

    # Test type promotion
    float32_group1 = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    float64_group2 = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    result_promoted = beignet.statistics.glass_delta(float32_group1, float64_group2)
    assert result_promoted.dtype == torch.float64

    # Test 1D input handling (scalar output)
    group1_1d = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    group2_1d = torch.tensor([2.0, 4.0], dtype=dtype)
    result_1d = beignet.statistics.glass_delta(group1_1d, group2_1d)
    assert result_1d.shape == ()  # Scalar output for 1D inputs

    # Manual calculation: mean1 = 2.0, mean2 = 3.0, std2 = sqrt(2) ≈ 1.414
    # Glass' delta = (2.0 - 3.0) / 1.414 ≈ -0.707
    expected_1d = (2.0 - 3.0) / torch.sqrt(torch.tensor(2.0, dtype=dtype))
    assert torch.allclose(result_1d, expected_1d, atol=1e-6)

    # Test difference from Cohen's d
    # Glass' delta uses only group2's variance, Cohen's d uses pooled variance
    cohens_d_result = beignet.statistics.cohens_d(group1, group2, pooled=True)
    glass_delta_result = beignet.statistics.glass_delta(group1, group2)

    # They should generally be different (unless group variances are very similar)
    # Don't assert inequality as they might coincidentally be close, just compute both
    assert cohens_d_result.shape == glass_delta_result.shape

    # Test batch consistency
    # Each batch element should follow the same formula independently
    batch_group1 = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=dtype,
    )
    batch_group2 = torch.tensor(
        [
            [0.0, 2.0],
            [3.0, 7.0],
            [6.0, 12.0],
        ],
        dtype=dtype,
    )

    batch_result = beignet.statistics.glass_delta(batch_group1, batch_group2)

    # Manually compute expected results for each batch element
    expected_batch = torch.zeros(3, dtype=dtype)
    for i in range(3):
        mean1 = torch.mean(batch_group1[i])
        mean2 = torch.mean(batch_group2[i])
        std2 = torch.sqrt(torch.var(batch_group2[i], correction=1))
        expected_batch[i] = (mean1 - mean2) / torch.clamp(
            std2,
            min=torch.finfo(dtype).eps,
        )

    assert torch.allclose(batch_result, expected_batch, atol=1e-6)

    # Test numerical stability with extreme values
    large_group1 = torch.tensor([[1e6, 1e6 + 1]], dtype=dtype)
    large_group2 = torch.tensor([[1e6, 1e6 + 2]], dtype=dtype)
    result_large = beignet.statistics.glass_delta(large_group1, large_group2)
    assert torch.all(torch.isfinite(result_large))

    # Test with mixed positive/negative values
    mixed_group1 = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=dtype)  # mean = 0
    mixed_group2 = torch.tensor([[-1.0, 0.0, 1.0]], dtype=dtype)  # mean = 0, std = 1
    result_mixed = beignet.statistics.glass_delta(mixed_group1, mixed_group2)
    # Glass' delta = (0.0 - 0.0) / 1.0 = 0.0
    assert torch.allclose(result_mixed, torch.tensor([0.0], dtype=dtype), atol=1e-6)

    # Test interpretation: Glass' delta measures effect size relative to control group variability
    # Larger control group variance should reduce Glass' delta magnitude
    treatment_group = torch.tensor([[5.0, 6.0, 7.0]], dtype=dtype)  # mean = 6

    low_var_control = torch.tensor([[3.0, 3.1]], dtype=dtype)  # mean ≈ 3.05, small std
    high_var_control = torch.tensor([[1.0, 5.0]], dtype=dtype)  # mean = 3.0, large std

    result_low_var = beignet.statistics.glass_delta(treatment_group, low_var_control)
    result_high_var = beignet.statistics.glass_delta(treatment_group, high_var_control)

    # Glass' delta should be larger in magnitude when control variance is smaller
    assert torch.abs(result_low_var) > torch.abs(result_high_var)

    # Test that Glass' delta preserves sign of mean difference
    positive_diff_group1 = torch.tensor([[4.0, 5.0, 6.0]], dtype=dtype)  # mean = 5
    negative_diff_group1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)  # mean = 2
    reference_group2 = torch.tensor([[3.0, 4.0]], dtype=dtype)  # mean = 3.5

    result_positive = beignet.statistics.glass_delta(
        positive_diff_group1,
        reference_group2,
    )
    result_negative = beignet.statistics.glass_delta(
        negative_diff_group1,
        reference_group2,
    )

    assert result_positive > 0  # 5 - 3.5 > 0
    assert result_negative < 0  # 2 - 3.5 < 0

    # Test correction parameter in variance calculation (Bessel's correction)
    # The implementation uses correction=1 (sample standard deviation)
    sample_group1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    sample_group2 = torch.tensor([[2.0, 4.0, 6.0]], dtype=dtype)

    # Manual calculation with correction=1
    mean1 = torch.mean(sample_group1)  # 2.0
    mean2 = torch.mean(sample_group2)  # 4.0
    var2_corrected = torch.var(sample_group2, correction=1)  # sample variance

    expected_corrected = (mean1 - mean2) / torch.sqrt(
        torch.clamp(var2_corrected, min=torch.finfo(dtype).eps),
    )
    result_corrected = beignet.statistics.glass_delta(sample_group1, sample_group2)

    assert torch.allclose(result_corrected, expected_corrected, atol=1e-6)
