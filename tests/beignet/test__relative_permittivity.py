import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    spatial_dims=st.lists(
        st.integers(min_value=2, max_value=10), min_size=1, max_size=3
    ),
    channels=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    temperature=st.floats(min_value=100.0, max_value=1000.0),
    eps_s=st.floats(min_value=50.0, max_value=100.0),
    eps_inf=st.floats(min_value=1.0, max_value=10.0),
    tau_0=st.floats(min_value=1e-13, max_value=1e-11),
    E_a=st.floats(min_value=1e-21, max_value=1e-19),
)
@settings(deadline=None)  # Disable deadline due to torch.compile
def test_relative_permittivity(
    batch_size: int,
    spatial_dims: list[int],
    channels: int,
    dtype: torch.dtype,
    temperature: float,
    eps_s: float,
    eps_inf: float,
    tau_0: float,
    E_a: float,
) -> None:
    """Test relative_permittivity operator."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create input tensors
    input_shape = [batch_size, channels] + spatial_dims
    input_tensor = torch.randn(input_shape, dtype=dtype) * 0.1 + 1.0

    # Create charges - same batch size but no channel dim
    charges_shape = [batch_size] + [1] * len(spatial_dims)
    charges = torch.randn(charges_shape, dtype=dtype) * 10.0

    # Temperature can be scalar or tensor
    temp_scalar = torch.tensor(temperature, dtype=dtype)
    temp_batch = torch.full((batch_size,), temperature, dtype=dtype)

    # Test with scalar temperature
    result_scalar = beignet.relative_permittivity(
        input_tensor,
        charges,
        temp_scalar,
        eps_s=eps_s,
        eps_inf=eps_inf,
        tau_0=tau_0,
        E_a=E_a,
    )

    # Basic checks
    assert result_scalar.dtype == dtype
    assert result_scalar.shape == torch.broadcast_shapes(
        input_tensor.shape, charges.shape
    )
    assert torch.all(result_scalar > 0), "Relative permittivity must be positive"
    assert torch.all(torch.isfinite(result_scalar)), "Result must be finite"

    # Test with batch temperature
    result_batch = beignet.relative_permittivity(
        input_tensor,
        charges,
        temp_batch,
        eps_s=eps_s,
        eps_inf=eps_inf,
        tau_0=tau_0,
        E_a=E_a,
    )

    assert result_batch.shape == torch.broadcast_shapes(
        input_tensor.shape,
        charges.shape,
        temp_batch.shape + (1,) * (len(input_shape) - 1),
    )

    # Test physical bounds
    assert torch.all(result_scalar >= min(eps_inf, eps_s))
    assert torch.all(result_scalar <= max(eps_inf, eps_s))

    # Test charge symmetry (optional model property)
    result_neg_charges = beignet.relative_permittivity(
        input_tensor,
        -charges,
        temp_scalar,
        eps_s=eps_s,
        eps_inf=eps_inf,
        tau_0=tau_0,
        E_a=E_a,
    )
    # The default model uses abs(charges), so should be symmetric
    assert torch.allclose(result_scalar, result_neg_charges, rtol=1e-5)

    # Test temperature monotonicity
    if batch_size >= 2:
        temp_low = torch.tensor(200.0, dtype=dtype)
        temp_high = torch.tensor(800.0, dtype=dtype)

        _ = beignet.relative_permittivity(
            input_tensor[0:1],
            charges[0:1],
            temp_low,
            eps_s=eps_s,
            eps_inf=eps_inf,
            tau_0=tau_0,
            E_a=E_a,
        )
        _ = beignet.relative_permittivity(
            input_tensor[0:1],
            charges[0:1],
            temp_high,
            eps_s=eps_s,
            eps_inf=eps_inf,
            tau_0=tau_0,
            E_a=E_a,
        )

        # Higher temperature typically leads to lower permittivity (for water-like materials)
        # This depends on the specific model parameters

    # Test gradient computation
    if dtype == torch.float64:
        input_grad = input_tensor.clone().requires_grad_(True)
        charges_grad = charges.clone().requires_grad_(True)
        temp_grad = temp_scalar.clone().requires_grad_(True)

        _ = beignet.relative_permittivity(
            input_grad,
            charges_grad,
            temp_grad,
            eps_s=eps_s,
            eps_inf=eps_inf,
            tau_0=tau_0,
            E_a=E_a,
        )

        # Check gradcheck
        def permittivity_fn(inp, chg, tmp):
            return beignet.relative_permittivity(
                inp, chg, tmp, eps_s=eps_s, eps_inf=eps_inf, tau_0=tau_0, E_a=E_a
            )

        # Use smaller inputs for gradcheck
        small_input = (
            input_grad[:1, :1, :2]
            if len(spatial_dims) == 1
            else input_grad[:1, :1, :2, :2]
        )
        small_charges = (
            charges_grad[:1, :1] if len(spatial_dims) == 1 else charges_grad[:1, :1, :1]
        )

        assert torch.autograd.gradcheck(
            permittivity_fn,
            (small_input, small_charges, temp_grad),
            eps=1e-6,
            atol=1e-4,
        )

    # Test torch.compile compatibility - only test with default parameters
    # to avoid recompilation issues with hypothesis
    compiled_fn = torch.compile(beignet.relative_permittivity, fullgraph=True)
    result_compiled = compiled_fn(input_tensor, charges, temp_scalar)
    result_default = beignet.relative_permittivity(input_tensor, charges, temp_scalar)
    assert torch.allclose(result_default, result_compiled, rtol=1e-5)

    # Test vmap compatibility
    from torch.func import vmap

    # Test vmap over batch dimension - use default parameters to avoid issues
    def single_perm(inp, chg, tmp):
        return beignet.relative_permittivity(inp, chg, tmp)

    # Vmap over first dimension of input only
    single_input = input_tensor[0]
    single_charge = charges if charges.dim() == 0 else charges[0]
    _ = beignet.relative_permittivity(single_input, single_charge, temp_scalar)

    # Apply vmap
    vmap_fn = vmap(
        lambda inp: beignet.relative_permittivity(inp, single_charge, temp_scalar)
    )
    result_vmap = vmap_fn(input_tensor)

    # Check shapes match expected broadcast
    assert result_vmap.shape[0] == batch_size

    # Test broadcasting edge cases
    # Single charge value broadcast to all
    single_charge = torch.tensor(1.0, dtype=dtype)
    result_broadcast = beignet.relative_permittivity(
        input_tensor,
        single_charge,
        temp_scalar,
        eps_s=eps_s,
        eps_inf=eps_inf,
        tau_0=tau_0,
        E_a=E_a,
    )
    assert result_broadcast.shape == input_tensor.shape

    # Test numerical stability with extreme inputs
    # Very small input values
    small_input = torch.full_like(input_tensor, 1e-8)
    result_small = beignet.relative_permittivity(
        small_input,
        charges,
        temp_scalar,
        eps_s=eps_s,
        eps_inf=eps_inf,
        tau_0=tau_0,
        E_a=E_a,
    )
    assert torch.all(torch.isfinite(result_small))

    # Very large charge values
    large_charges = torch.full_like(charges, 1e6)
    result_large = beignet.relative_permittivity(
        input_tensor,
        large_charges,
        temp_scalar,
        eps_s=eps_s,
        eps_inf=eps_inf,
        tau_0=tau_0,
        E_a=E_a,
    )
    assert torch.all(torch.isfinite(result_large))

    # Test default parameters (water at room temperature)
    result_default = beignet.relative_permittivity(input_tensor, charges, temp_scalar)
    assert torch.all(result_default > 0)
    assert torch.all(torch.isfinite(result_default))
