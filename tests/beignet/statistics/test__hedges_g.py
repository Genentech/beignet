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
def test_hedges_g(batch_size, n1, n2, dtype):
    """Test Hedges' g effect size calculation."""
    # Generate test data
    group1 = torch.randn(batch_size, n1, dtype=dtype)
    group2 = torch.randn(batch_size, n2, dtype=dtype)

    # Test basic functionality
    result = beignet.statistics.hedges_g(group1, group2)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype

    # Test with identical groups (should be close to zero)
    identical_group = torch.randn(batch_size, n1, dtype=dtype)
    result_identical = beignet.statistics.hedges_g(identical_group, identical_group)
    assert torch.allclose(
        result_identical,
        torch.zeros_like(result_identical),
        atol=1e-6,
    )

    # Test symmetry property: Hedges' g(A, B) = -Hedges' g(B, A)
    result_forward = beignet.statistics.hedges_g(group1, group2)
    result_backward = beignet.statistics.hedges_g(group2, group1)
    assert torch.allclose(result_forward, -result_backward, atol=1e-6)

    # Test with out parameter
    out = torch.empty(batch_size, dtype=dtype)
    result_out = beignet.statistics.hedges_g(group1, group2, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test relationship with Cohen's d
    cohens_d_result = beignet.statistics.cohens_d(group1, group2, pooled=True)
    hedges_g_result = beignet.statistics.hedges_g(group1, group2)

    # Hedges' g should be smaller in magnitude than Cohen's d (bias correction)
    assert torch.all(torch.abs(hedges_g_result) <= torch.abs(cohens_d_result))

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)
    result_grad = beignet.statistics.hedges_g(group1_grad, group2_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test torch.compile compatibility
    compiled_hedges_g = torch.compile(beignet.statistics.hedges_g, fullgraph=True)
    result_compiled = compiled_hedges_g(group1, group2)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test bias correction factor
    # For large sample sizes, Hedges' g should approach Cohen's d
    large_n1, large_n2 = 100, 100
    large_group1 = torch.randn(1, large_n1, dtype=dtype)
    large_group2 = torch.randn(1, large_n2, dtype=dtype)

    large_cohens_d = beignet.statistics.cohens_d(
        large_group1,
        large_group2,
        pooled=True,
    )
    large_hedges_g = beignet.statistics.hedges_g(large_group1, large_group2)

    # Should be very close for large samples
    assert torch.allclose(large_cohens_d, large_hedges_g, atol=0.01)

    # For small sample sizes, bias correction should be more pronounced
    small_n1, small_n2 = 5, 5
    small_group1 = torch.randn(1, small_n1, dtype=dtype)
    small_group2 = torch.randn(1, small_n2, dtype=dtype) + 1.0  # Add effect

    small_cohens_d = beignet.statistics.cohens_d(
        small_group1,
        small_group2,
        pooled=True,
    )
    small_hedges_g = beignet.statistics.hedges_g(small_group1, small_group2)

    # Hedges' g should be smaller in magnitude due to bias correction
    assert torch.abs(small_hedges_g) < torch.abs(small_cohens_d)

    # Test mathematical properties
    # For known distributions with effect size
    torch.manual_seed(42)
    normal1 = torch.normal(0.0, 1.0, size=(1, 50), dtype=dtype)
    normal2 = torch.normal(
        1.0,
        1.0,
        size=(1, 50),
        dtype=dtype,
    )  # Effect size should be ~1.0

    result_known = beignet.statistics.hedges_g(normal1, normal2)
    # Should be approximately -1.0 with bias correction (negative because normal2 > normal1)
    assert (
        torch.abs(result_known + 1.0) < 0.3
    )  # Allow variance due to sampling and bias correction

    # Test explicit bias correction formula verification
    # For deterministic case: verify J(df) = 1 - 3/(4*df - 1)
    test_n1, test_n2 = 10, 12
    test_group1 = torch.zeros(1, test_n1, dtype=dtype)
    test_group2 = torch.ones(1, test_n2, dtype=dtype)

    df = test_n1 + test_n2 - 2  # 20
    expected_correction = 1.0 - 3.0 / (4.0 * df - 1.0)  # 1 - 3/79 ≈ 0.962

    cohens_d_manual = beignet.statistics.cohens_d(test_group1, test_group2, pooled=True)
    hedges_g_manual = beignet.statistics.hedges_g(test_group1, test_group2)
    expected_hedges_g = cohens_d_manual * expected_correction

    assert torch.allclose(hedges_g_manual, expected_hedges_g, atol=1e-6)

    # Test limiting behavior: as sample size increases, correction approaches 1
    very_large_n = 1000
    large_group1_test = torch.zeros(1, very_large_n, dtype=dtype)
    large_group2_test = torch.ones(1, very_large_n, dtype=dtype)

    large_df = 2 * very_large_n - 2
    large_correction = 1.0 - 3.0 / (4.0 * large_df - 1.0)

    # Correction should be very close to 1.0 for large samples
    assert large_correction > 0.999

    large_cohens_d_test = beignet.statistics.cohens_d(
        large_group1_test,
        large_group2_test,
        pooled=True,
    )
    large_hedges_g_test = beignet.statistics.hedges_g(
        large_group1_test,
        large_group2_test,
    )

    # Should be nearly identical for very large samples
    assert torch.allclose(large_cohens_d_test, large_hedges_g_test, atol=1e-3)

    # Test minimum sample size behavior (n=2 each, df=2)
    min_group1 = torch.tensor([[0.0, 1.0]], dtype=dtype)
    min_group2 = torch.tensor([[2.0, 3.0]], dtype=dtype)

    min_df = 2
    min_correction = 1.0 - 3.0 / (4.0 * min_df - 1.0)  # 1 - 3/7 ≈ 0.571

    min_cohens_d = beignet.statistics.cohens_d(min_group1, min_group2, pooled=True)
    min_hedges_g = beignet.statistics.hedges_g(min_group1, min_group2)
    expected_min_hedges_g = min_cohens_d * min_correction

    assert torch.allclose(min_hedges_g, expected_min_hedges_g, atol=1e-6)

    # Verify substantial bias correction for small samples
    assert (
        torch.abs(min_hedges_g) < torch.abs(min_cohens_d) * 0.8
    )  # At least 20% reduction

    # Test edge case: equal sample sizes
    equal_n = 15
    equal_group1 = torch.randn(1, equal_n, dtype=dtype)
    equal_group2 = torch.randn(1, equal_n, dtype=dtype) + 0.5

    equal_df = 2 * equal_n - 2
    equal_correction = 1.0 - 3.0 / (4.0 * equal_df - 1.0)

    equal_cohens_d = beignet.statistics.cohens_d(
        equal_group1,
        equal_group2,
        pooled=True,
    )
    equal_hedges_g = beignet.statistics.hedges_g(equal_group1, equal_group2)
    expected_equal_hedges_g = equal_cohens_d * equal_correction

    assert torch.allclose(equal_hedges_g, expected_equal_hedges_g, atol=1e-6)

    # Test unequal sample sizes
    unequal_group1 = torch.randn(1, 8, dtype=dtype)
    unequal_group2 = torch.randn(1, 25, dtype=dtype) + 0.3

    unequal_df = 8 + 25 - 2  # 31
    unequal_correction = 1.0 - 3.0 / (4.0 * unequal_df - 1.0)

    unequal_cohens_d = beignet.statistics.cohens_d(
        unequal_group1,
        unequal_group2,
        pooled=True,
    )
    unequal_hedges_g = beignet.statistics.hedges_g(unequal_group1, unequal_group2)
    expected_unequal_hedges_g = unequal_cohens_d * unequal_correction

    assert torch.allclose(unequal_hedges_g, expected_unequal_hedges_g, atol=1e-6)

    # Test correction factor bounds: should always be between 0 and 1
    for test_n1_bound in [3, 5, 10, 50, 100]:
        for test_n2_bound in [3, 5, 10, 50, 100]:
            df_bound = test_n1_bound + test_n2_bound - 2
            correction_bound = 1.0 - 3.0 / (4.0 * df_bound - 1.0)
            assert 0.0 < correction_bound < 1.0, (
                f"Correction {correction_bound} out of bounds for df={df_bound}"
            )

    # Test numerical properties
    # Hedges' g should preserve the sign of Cohen's d
    positive_effect_group1 = torch.zeros(1, 10, dtype=dtype)
    positive_effect_group2 = torch.ones(1, 10, dtype=dtype)

    pos_cohens_d = beignet.statistics.cohens_d(
        positive_effect_group1,
        positive_effect_group2,
        pooled=True,
    )
    pos_hedges_g = beignet.statistics.hedges_g(
        positive_effect_group1,
        positive_effect_group2,
    )

    # Should have same sign
    assert torch.sign(pos_cohens_d) == torch.sign(pos_hedges_g)

    # Test with zero variance (edge case handling)
    zero_var_group1 = torch.ones(1, 10, dtype=dtype) * 5.0  # Constant
    zero_var_group2 = torch.ones(1, 10, dtype=dtype) * 7.0  # Different constant

    zero_var_hedges_g = beignet.statistics.hedges_g(zero_var_group1, zero_var_group2)
    # Should handle gracefully (likely return inf or large value, but finite)
    assert torch.all(torch.isfinite(zero_var_hedges_g) | torch.isinf(zero_var_hedges_g))

    # Test batch consistency: each batch element should follow the same correction
    batch_group1 = torch.randn(3, 8, dtype=dtype)
    batch_group2 = torch.randn(3, 12, dtype=dtype)

    batch_cohens_d = beignet.statistics.cohens_d(
        batch_group1,
        batch_group2,
        pooled=True,
    )
    batch_hedges_g = beignet.statistics.hedges_g(batch_group1, batch_group2)

    # All batch elements should have same correction factor
    batch_df = 8 + 12 - 2
    batch_correction = 1.0 - 3.0 / (4.0 * batch_df - 1.0)
    expected_batch_hedges_g = batch_cohens_d * batch_correction

    assert torch.allclose(batch_hedges_g, expected_batch_hedges_g, atol=1e-6)
