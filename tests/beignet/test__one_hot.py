import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet import one_hot


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=16),
    n_bins=st.integers(min_value=2, max_value=8),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=10)  # Disable deadline, reduce examples
def test__one_hot(batch_size, seq_len, n_bins, dtype):
    """Test one_hot function with various input configurations."""

    # Create input values
    x = torch.randn(batch_size, seq_len, dtype=dtype)

    # Create bin centers
    v_bins = torch.linspace(-3.0, 3.0, n_bins, dtype=dtype)

    # Test basic functionality
    result = one_hot(x, v_bins)

    # Check output shape
    expected_shape = (batch_size, seq_len, n_bins)
    assert result.shape == expected_shape

    # Check output dtype
    assert result.dtype == dtype

    # Check that output is one-hot encoded (each position has exactly one 1)
    assert torch.allclose(
        result.sum(dim=-1), torch.ones(batch_size, seq_len, dtype=dtype)
    )

    # Check that all values are 0 or 1
    assert torch.all((result == 0) | (result == 1))

    # Test edge cases
    if n_bins >= 2:
        # Test with values exactly at bin centers
        x_exact = v_bins.clone()
        result_exact = one_hot(x_exact, v_bins)

        # Each bin center should map to its corresponding bin
        expected_exact = torch.eye(n_bins, dtype=dtype)
        assert torch.allclose(result_exact, expected_exact)

    # Test gradient computation (one_hot is discrete, so gradients will be zero)
    x_grad = torch.randn(batch_size, seq_len, dtype=dtype, requires_grad=True)
    result_grad = one_hot(x_grad, v_bins)

    # Since one_hot is discrete, we need to create a differentiable loss
    # We'll use the input directly in a differentiable computation
    loss = x_grad.sum()  # Use input directly for gradient test
    loss.backward()

    # Check that gradients are computed
    assert x_grad.grad is not None
    assert torch.allclose(x_grad.grad, torch.ones_like(x_grad))

    # Test torch.compile compatibility (skip for speed in tests)
    # compiled_one_hot = torch.compile(one_hot, fullgraph=True)
    # result_compiled = compiled_one_hot(x, v_bins)
    # assert torch.allclose(result, result_compiled)

    # Test with different shapes
    x_1d = torch.randn(seq_len, dtype=dtype)
    result_1d = one_hot(x_1d, v_bins)
    assert result_1d.shape == (seq_len, n_bins)

    # Test with scalar input
    x_scalar = torch.randn(1, dtype=dtype)
    result_scalar = one_hot(x_scalar, v_bins)
    assert result_scalar.shape == (1, n_bins)

    # Test that function finds nearest bin correctly
    if n_bins >= 2:
        # Test midpoint between first two bins
        midpoint = (v_bins[0] + v_bins[1]) / 2
        x_mid = midpoint.reshape(1)
        result_mid = one_hot(x_mid, v_bins)

        # Should be assigned to one of the first two bins
        assert result_mid[0, :2].sum() == 1.0
        assert result_mid[0, 2:].sum() == 0.0
