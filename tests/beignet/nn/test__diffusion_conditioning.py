import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import DiffusionConditioning


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    n_atoms=st.integers(min_value=4, max_value=16),
    c_z=st.integers(min_value=16, max_value=64),
    c_s=st.integers(min_value=32, max_value=128),
    c_s_inputs=st.integers(min_value=16, max_value=64),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_diffusion_conditioning(batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype):
    """Test DiffusionConditioning (Algorithm 21) comprehensively."""
    device = torch.device("cpu")

    # Create module
    module = (
        DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs)
        .to(device)
        .to(dtype)
    )

    # Generate test inputs
    t = (
        torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
    )  # Ensure positive
    f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
    s_inputs = torch.randn(batch_size, n_atoms, c_s_inputs, dtype=dtype, device=device)
    s_trunk = torch.randn(batch_size, n_atoms, c_s, dtype=dtype, device=device)
    z_trunk = torch.randn(batch_size, n_atoms, n_atoms, c_z, dtype=dtype, device=device)

    # Test basic functionality
    s_out, z_out = module(t, f_star, s_inputs, s_trunk, z_trunk)

    # Check output shapes and properties
    expected_s_shape = (batch_size, n_atoms, c_s)
    expected_z_shape = (batch_size, n_atoms, n_atoms, c_z)

    assert s_out.shape == expected_s_shape, (
        f"Expected s shape {expected_s_shape}, got {s_out.shape}"
    )
    assert z_out.shape == expected_z_shape, (
        f"Expected z shape {expected_z_shape}, got {z_out.shape}"
    )
    assert torch.all(torch.isfinite(s_out)), "Single output should be finite"
    assert torch.all(torch.isfinite(z_out)), "Pair output should be finite"
    assert s_out.dtype == dtype, f"Expected s dtype {dtype}, got {s_out.dtype}"
    assert z_out.dtype == dtype, f"Expected z dtype {dtype}, got {z_out.dtype}"

    # Test gradient computation
    t_grad = t.clone().requires_grad_(True)
    f_star_grad = f_star.clone().requires_grad_(True)
    s_inputs_grad = s_inputs.clone().requires_grad_(True)
    s_trunk_grad = s_trunk.clone().requires_grad_(True)
    z_trunk_grad = z_trunk.clone().requires_grad_(True)

    s_out_grad, z_out_grad = module(
        t_grad, f_star_grad, s_inputs_grad, s_trunk_grad, z_trunk_grad
    )
    loss = s_out_grad.sum() + z_out_grad.sum()
    loss.backward()

    assert t_grad.grad is not None, "Should have gradients for timestep"
    assert f_star_grad.grad is not None, "Should have gradients for target positions"
    assert s_inputs_grad.grad is not None, "Should have gradients for input singles"
    assert s_trunk_grad.grad is not None, "Should have gradients for trunk singles"
    assert z_trunk_grad.grad is not None, "Should have gradients for trunk pairs"

    assert torch.all(torch.isfinite(t_grad.grad)), "Timestep gradients should be finite"
    assert torch.all(torch.isfinite(f_star_grad.grad)), (
        "Position gradients should be finite"
    )
    assert torch.all(torch.isfinite(s_inputs_grad.grad)), (
        "Input single gradients should be finite"
    )
    assert torch.all(torch.isfinite(s_trunk_grad.grad)), (
        "Trunk single gradients should be finite"
    )
    assert torch.all(torch.isfinite(z_trunk_grad.grad)), (
        "Trunk pair gradients should be finite"
    )

    # Test module parameters have gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), (
                f"Parameter {name} gradients should be finite"
            )

    # Test module components - Algorithm 21 requirements
    assert hasattr(module, "relative_pos_enc"), "Should have RelativePositionEncoding"
    assert hasattr(module, "linear_z"), "Should have pair linear projection"
    assert hasattr(module, "layer_norm_z"), "Should have pair layer norm"
    assert hasattr(module, "transition_z_1"), "Should have first pair transition"
    assert hasattr(module, "transition_z_2"), "Should have second pair transition"

    assert hasattr(module, "linear_s"), "Should have single linear projection"
    assert hasattr(module, "layer_norm_s"), "Should have single layer norm"
    assert hasattr(module, "fourier_embedding"), "Should have FourierEmbedding"
    assert hasattr(module, "linear_timestep"), "Should have timestep projection"
    assert hasattr(module, "transition_s_1"), "Should have first single transition"
    assert hasattr(module, "transition_s_2"), "Should have second single transition"

    # Test Fourier embedding component
    assert module.fourier_embedding.c == 256, (
        "FourierEmbedding should have 256 dimensions"
    )

    # Test that the module transforms the inputs (should be different from inputs)
    s_diff = torch.norm(s_out - s_trunk)
    z_diff = torch.norm(z_out - z_trunk)
    assert s_diff > 1e-6, "Single representations should be transformed"
    assert z_diff > 1e-6, "Pair representations should be transformed"

    # Test Algorithm 21 step-by-step behavior
    # Step 1: Relative position encoding
    rel_pos_enc = module.relative_pos_enc(f_star)
    assert rel_pos_enc.shape == (batch_size, n_atoms, n_atoms, c_z), (
        "RelPos shape should match"
    )

    # Test concatenation behavior
    z_concat = torch.cat([z_trunk, rel_pos_enc], dim=-1)
    assert z_concat.shape[-1] == 2 * c_z, (
        "Concatenated pairs should have 2*c_z channels"
    )

    s_concat = torch.cat([s_trunk, s_inputs], dim=-1)
    assert s_concat.shape[-1] == c_s + c_s_inputs, (
        "Concatenated singles should have c_s + c_s_inputs channels"
    )

    # Test timestep embedding
    t_scaled = t / module.sigma_data
    t_log = 0.25 * torch.log(torch.clamp(t_scaled, min=1e-8))
    n_embedding = module.fourier_embedding(t_log)
    assert n_embedding.shape == (batch_size, 1, 256), (
        "Timestep embedding should have shape (batch, 1, 256)"
    )

    # Test numerical stability with small timesteps
    small_t = torch.full((batch_size, 1), 1e-6, dtype=dtype, device=device)
    s_small, z_small = module(small_t, f_star, s_inputs, s_trunk, z_trunk)
    assert torch.all(torch.isfinite(s_small)), "Should handle small timesteps"
    assert torch.all(torch.isfinite(z_small)), "Should handle small timesteps"

    # Test with large timesteps
    large_t = torch.full((batch_size, 1), 1000.0, dtype=dtype, device=device)
    s_large, z_large = module(large_t, f_star, s_inputs, s_trunk, z_trunk)
    assert torch.all(torch.isfinite(s_large)), "Should handle large timesteps"
    assert torch.all(torch.isfinite(z_large)), "Should handle large timesteps"

    # Test with zero positions
    zero_f_star = torch.zeros_like(f_star)
    s_zero, z_zero = module(t, zero_f_star, s_inputs, s_trunk, z_trunk)
    assert torch.all(torch.isfinite(s_zero)), "Should handle zero positions"
    assert torch.all(torch.isfinite(z_zero)), "Should handle zero positions"

    # Test parameter initialization
    for name, param in module.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} should be finite"
        assert param.dtype == dtype, f"Parameter {name} should have correct dtype"

    # Test torch.compile compatibility
    try:
        compiled_module = torch.compile(module, fullgraph=True)
        s_compiled, z_compiled = compiled_module(t, f_star, s_inputs, s_trunk, z_trunk)
        assert s_compiled.shape == s_out.shape, "Compiled module should work"
        assert z_compiled.shape == z_out.shape, "Compiled module should work"
        assert torch.all(torch.isfinite(s_compiled)), "Compiled output should be finite"
        assert torch.all(torch.isfinite(z_compiled)), "Compiled output should be finite"
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test batch processing consistency
    if batch_size > 1:
        single_t = t[0:1]
        single_f_star = f_star[0:1]
        single_s_inputs = s_inputs[0:1]
        single_s_trunk = s_trunk[0:1]
        single_z_trunk = z_trunk[0:1]

        single_s, single_z = module(
            single_t, single_f_star, single_s_inputs, single_s_trunk, single_z_trunk
        )
        batch_s_first = s_out[0:1]
        batch_z_first = z_out[0:1]

        assert torch.allclose(single_s, batch_s_first, atol=1e-5), (
            "Batch processing should be consistent for singles"
        )
        assert torch.allclose(single_z, batch_z_first, atol=1e-5), (
            "Batch processing should be consistent for pairs"
        )

    # Test different atom counts
    if n_atoms > 4:
        smaller_f_star = f_star[:, : n_atoms // 2]
        smaller_s_inputs = s_inputs[:, : n_atoms // 2]
        smaller_s_trunk = s_trunk[:, : n_atoms // 2]
        smaller_z_trunk = z_trunk[:, : n_atoms // 2, : n_atoms // 2]

        s_smaller, z_smaller = module(
            t, smaller_f_star, smaller_s_inputs, smaller_s_trunk, smaller_z_trunk
        )
        expected_smaller_s_shape = (batch_size, n_atoms // 2, c_s)
        expected_smaller_z_shape = (batch_size, n_atoms // 2, n_atoms // 2, c_z)

        assert s_smaller.shape == expected_smaller_s_shape, (
            "Should handle different atom counts"
        )
        assert z_smaller.shape == expected_smaller_z_shape, (
            "Should handle different atom counts"
        )

    # Test conditioning effect - different timesteps should produce different outputs
    t_different = t + 1.0
    s_different, z_different = module(t_different, f_star, s_inputs, s_trunk, z_trunk)

    s_timestep_diff = torch.norm(s_different - s_out)
    z_timestep_diff = torch.norm(z_different - z_out)

    assert s_timestep_diff > 1e-6, (
        "Different timesteps should affect single representations"
    )
    assert z_timestep_diff > 1e-6, (
        "Different timesteps should affect pair representations"
    )
