import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    power=st.floats(min_value=0.1, max_value=0.95),
    groups=st.integers(min_value=3, max_value=8),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_analysis_of_variance_sample_size(
    batch_size,
    effect_size,
    power,
    groups,
    alpha,
    dtype,
):
    """Test analysis_of_variance_sample_size functional wrapper."""
    # Create test inputs
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    groups_tensor = torch.full((batch_size,), groups, dtype=torch.int64)

    # Test functional wrapper - correct signature: (effect_size, groups, power=float, alpha=float)
    result_functional = (
        beignet.metrics.functional.statistics.analysis_of_variance_sample_size(
            effect_size_tensor,
            groups_tensor,
            power=power,
            alpha=alpha,
        )
    )

    # Test direct call to beignet.statistics
    result_direct = beignet.statistics.analysis_of_variance_sample_size(
        effect_size_tensor,
        groups_tensor,
        power=power,
        alpha=alpha,
    )

    # Results should be identical - handle edge cases where both might be NaN/inf
    if torch.all(torch.isfinite(result_functional)) and torch.all(
        torch.isfinite(result_direct),
    ):
        assert torch.allclose(result_functional, result_direct, atol=1e-6)
    else:
        # Both should have the same finite/non-finite pattern
        assert torch.equal(
            torch.isfinite(result_functional),
            torch.isfinite(result_direct),
        )

    # Verify output properties
    assert isinstance(result_functional, torch.Tensor)
    assert result_functional.shape == (batch_size,)
    assert result_functional.dtype == dtype

    # Only check minimum sample size if results are finite
    if torch.all(torch.isfinite(result_functional)):
        assert torch.all(result_functional >= 3.0)  # Should be at least 3 for ANOVA

    # Test parameter passing - different alpha should give different results (skip if sample size is minimal)
    if alpha * 2 <= 0.1:  # Only test if alpha is valid
        result_different_alpha = (
            beignet.metrics.functional.statistics.analysis_of_variance_sample_size(
                effect_size_tensor,
                groups_tensor,
                power=power,
                alpha=alpha * 2,
            )
        )
        # Only test difference if results are finite and sample size is meaningful (not minimal)
        if (
            torch.all(torch.isfinite(result_functional))
            and torch.all(torch.isfinite(result_different_alpha))
            and torch.all(result_functional > 10)
        ):  # Only test when sample size is not at minimum
            assert not torch.allclose(
                result_functional,
                result_different_alpha,
                atol=1e-6,
            )

    # Test gradient computation (only if results are finite)
    if torch.all(torch.isfinite(result_functional)):
        effect_grad = effect_size_tensor.clone().requires_grad_(True)
        groups_grad = groups_tensor.clone().float().requires_grad_(True)

        result_grad = (
            beignet.metrics.functional.statistics.analysis_of_variance_sample_size(
                effect_grad,
                groups_grad,
                power=power,
                alpha=alpha,
            )
        )

        if torch.all(torch.isfinite(result_grad)):
            loss = result_grad.sum()
            loss.backward()

            assert effect_grad.grad is not None
            assert groups_grad.grad is not None

    # Test edge cases - verify function doesn't crash with extreme values
    small_effect = torch.full((batch_size,), 0.1, dtype=dtype)
    size_small = beignet.metrics.functional.statistics.analysis_of_variance_sample_size(
        small_effect,
        groups_tensor,
        power=power,
        alpha=alpha,
    )

    large_effect = torch.full((batch_size,), 1.5, dtype=dtype)
    size_large = beignet.metrics.functional.statistics.analysis_of_variance_sample_size(
        large_effect,
        groups_tensor,
        power=power,
        alpha=alpha,
    )

    # Only compare if both results are finite (avoid comparing NaN or inf)
    if torch.all(torch.isfinite(size_small)) and torch.all(torch.isfinite(size_large)):
        assert torch.all(
            size_small >= size_large,
        )  # Smaller effect should need same or more samples
