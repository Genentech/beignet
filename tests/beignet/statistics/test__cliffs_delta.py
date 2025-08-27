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
def test_cliffs_delta(batch_size, n1, n2, dtype):
    """Test Cliff's Delta effect size calculation."""
    # Generate test data
    group1 = torch.randn(batch_size, n1, dtype=dtype)
    group2 = torch.randn(batch_size, n2, dtype=dtype)

    # Test basic functionality
    result = beignet.statistics.cliffs_delta(group1, group2)
    assert result.shape == (batch_size,)
    assert result.dtype == torch.promote_types(group1.dtype, group2.dtype)

    # Test that result is bounded between -1 and 1
    assert torch.all(result >= -1.0)
    assert torch.all(result <= 1.0)

    # Test with identical groups (should be close to zero)
    identical_group = torch.randn(batch_size, n1, dtype=dtype)
    result_identical = beignet.statistics.cliffs_delta(
        identical_group,
        identical_group,
    )
    assert torch.allclose(
        result_identical,
        torch.zeros_like(result_identical),
        atol=1e-6,
    )

    # Test anti-symmetry property: Cliff's δ(A, B) = -Cliff's δ(B, A)
    result_forward = beignet.statistics.cliffs_delta(group1, group2)
    result_backward = beignet.statistics.cliffs_delta(group2, group1)
    assert torch.allclose(result_forward, -result_backward, atol=1e-6)

    # Test with out parameter
    out = torch.empty_like(result)
    result_out = beignet.statistics.cliffs_delta(group1, group2, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)
    result_grad = beignet.statistics.cliffs_delta(group1_grad, group2_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test torch.compile compatibility
    compiled_cliffs_delta = torch.compile(
        beignet.statistics.cliffs_delta,
        fullgraph=True,
    )
    result_compiled = compiled_cliffs_delta(group1, group2)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test with extreme cases
    # Group 1 all zeros, Group 2 all ones -> should be close to -1
    zeros_group = torch.zeros(1, 10, dtype=dtype)
    ones_group = torch.ones(1, 10, dtype=dtype)
    result_extreme = beignet.statistics.cliffs_delta(zeros_group, ones_group)
    assert torch.all(result_extreme < -0.9)  # Should be very negative

    # Group 1 all ones, Group 2 all zeros -> should be close to 1
    result_extreme_reverse = beignet.statistics.cliffs_delta(ones_group, zeros_group)
    assert torch.all(result_extreme_reverse > 0.9)  # Should be very positive

    # Test with single values (1D tensors become scalar output)
    single1 = torch.tensor([1.0], dtype=dtype)
    single2 = torch.tensor([2.0], dtype=dtype)
    result_single = beignet.statistics.cliffs_delta(single1, single2)
    assert result_single.shape == ()  # Scalar output
    assert torch.allclose(result_single, torch.tensor(-1.0, dtype=dtype))

    # Test known ordinal dominance pattern
    # Create non-overlapping groups where group2 values are all > group1 values
    torch.manual_seed(42)
    low_group = torch.rand(1, 15, dtype=dtype) * 10.0  # [0, 10]
    high_group = torch.rand(1, 15, dtype=dtype) * 10.0 + 15.0  # [15, 25]

    result_ordinal = beignet.statistics.cliffs_delta(low_group, high_group)
    # Should be close to -1 since all values in high_group > low_group
    assert torch.all(result_ordinal < -0.95)

    # Test edge case: one group much larger than the other
    large_group = torch.randn(1, 100, dtype=dtype)
    small_group = torch.randn(1, 5, dtype=dtype)
    result_unbalanced = beignet.statistics.cliffs_delta(large_group, small_group)
    assert result_unbalanced.shape == (1,)
    assert torch.all(result_unbalanced >= -1.0)
    assert torch.all(result_unbalanced <= 1.0)

    # Test type promotion
    float32_group = torch.randn(1, 10, dtype=torch.float32)
    float64_group = torch.randn(1, 10, dtype=torch.float64)
    result_promoted = beignet.statistics.cliffs_delta(float32_group, float64_group)
    assert result_promoted.dtype == torch.float64

    # Test 1D input handling (becomes scalar output)
    group1_1d = torch.randn(10, dtype=dtype)
    group2_1d = torch.randn(12, dtype=dtype)
    result_1d = beignet.statistics.cliffs_delta(group1_1d, group2_1d)
    assert result_1d.shape == ()  # Scalar output for 1D inputs

    # Test mathematical correctness with known values
    # Create deterministic case
    group_a = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    group_b = torch.tensor([[4.0, 5.0]], dtype=dtype)

    # All values in group_b > all values in group_a
    # So P(X > Y) = 1.0, P(Y > X) = 0.0
    # Cliff's δ = (1.0 - 0.0) / (3 * 2) * (3 * 2) = 6/6 = 1.0...
    # Wait, let me recalculate: δ = (# X > Y - # Y > X) / (n1 * n2)
    # # X > Y = 0 (since group_a < group_b), # Y > X = 6
    # δ = (0 - 6) / 6 = -1.0
    result_known = beignet.statistics.cliffs_delta(group_a, group_b)
    assert torch.allclose(result_known, torch.tensor([-1.0], dtype=dtype), atol=1e-6)
