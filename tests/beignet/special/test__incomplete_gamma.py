import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.special


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_incomplete_gamma(batch_size, dtype):
    """Test regularized incomplete gamma function."""

    # Generate test parameters
    a_values = (
        torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0], dtype=dtype)
        .repeat(batch_size, 1)
        .flatten()
    )
    x_values = (
        torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], dtype=dtype)
        .repeat(batch_size, 1)
        .flatten()
    )

    # Test basic functionality
    result = beignet.special.incomplete_gamma(a_values, x_values)
    assert result.shape == a_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(a_values)
    result_out = beignet.special.incomplete_gamma(a_values, x_values, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test against torch.special.gammainc for accuracy
    torch_result = torch.special.gammainc(a_values, x_values)
    max_diff = torch.max(torch.abs(result - torch_result))
    assert max_diff < 1e-5, f"Maximum difference: {max_diff}"

    # Test boundary conditions
    # igamma(a, 0) should be approximately 0
    zero_x = torch.zeros_like(a_values)
    result_zero = beignet.special.incomplete_gamma(a_values, zero_x)
    assert torch.allclose(result_zero, torch.zeros_like(result_zero), atol=1e-6)

    # Test gradient computation for x
    x_grad = torch.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    a_const = torch.tensor([2.0, 3.0], dtype=dtype)
    result_grad = beignet.special.incomplete_gamma(a_const, x_grad)
    loss = result_grad.sum()
    loss.backward()

    # Gradient should exist and be finite
    assert x_grad.grad is not None
    assert torch.all(torch.isfinite(x_grad.grad))

    # Test torch.compile compatibility
    compiled_igamma = torch.compile(beignet.special.incomplete_gamma, fullgraph=True)
    a_simple = torch.tensor([2.0], dtype=dtype)
    x_simple = torch.tensor([1.0], dtype=dtype)
    result_compiled = compiled_igamma(a_simple, x_simple)
    result_normal = beignet.special.incomplete_gamma(a_simple, x_simple)
    assert torch.allclose(result_compiled, result_normal)
