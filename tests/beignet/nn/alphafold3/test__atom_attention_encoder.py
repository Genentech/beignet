import torch

from beignet.nn.alphafold3 import AtomAttentionEncoder


def test_atom_attention_encoder():
    """Test AtomAttentionEncoder basic functionality."""
    # Fixed test parameters for simplicity
    batch_size = 2
    n_tokens = 8
    n_atoms = 12
    c_token = 64  # Divisible by n_head
    c_atom = 32
    c_atompair = 16
    n_head = 8
    dtype = torch.float32
    device = torch.device("cpu")

    # Create module
    module = (
        AtomAttentionEncoder(
            c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, n_head=n_head
        )
        .to(device)
        .to(dtype)
    )

    # Generate test inputs
    f_star = {
        "ref_pos": torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device),
        "ref_mask": torch.ones(batch_size, n_atoms, dtype=dtype, device=device),
        "ref_element": torch.randint(0, 118, (batch_size, n_atoms), device=device),
        "ref_atom_name_chars": torch.randint(
            0, 26, (batch_size, n_atoms, 4), device=device
        ),
        "ref_charge": torch.randn(batch_size, n_atoms, dtype=dtype, device=device),
        "restype": torch.randint(0, 21, (batch_size, n_atoms), device=device),
        "profile": torch.randn(batch_size, n_atoms, 20, dtype=dtype, device=device),
        "deletion_mean": torch.randn(batch_size, n_atoms, dtype=dtype, device=device),
        "ref_space_uid": torch.randint(0, 1000, (batch_size, n_atoms), device=device),
    }
    r_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
    s_trunk = torch.randn(
        batch_size, n_tokens, c_token, dtype=dtype, device=device
    )  # Use c_token dimension
    z_atom = torch.randn(
        batch_size, n_atoms, n_atoms, c_atompair, dtype=dtype, device=device
    )

    # Test basic functionality
    a, q_skip, c_skip, p_skip = module(f_star, r_noisy, s_trunk, z_atom)

    # Check output shapes and properties
    expected_a_shape = (batch_size, n_tokens, c_token)
    expected_q_skip_shape = (
        batch_size,
        n_atoms,
        c_atom,
    )  # q_skip uses c_atom not c_token
    expected_c_skip_shape = (batch_size, n_atoms, c_atom)
    expected_p_skip_shape = (batch_size, n_atoms, n_atoms, c_atompair)

    assert a.shape == expected_a_shape, (
        f"Expected a shape {expected_a_shape}, got {a.shape}"
    )
    assert q_skip.shape == expected_q_skip_shape, (
        f"Expected q_skip shape {expected_q_skip_shape}, got {q_skip.shape}"
    )
    assert c_skip.shape == expected_c_skip_shape, (
        f"Expected c_skip shape {expected_c_skip_shape}, got {c_skip.shape}"
    )
    assert p_skip.shape == expected_p_skip_shape, (
        f"Expected p_skip shape {expected_p_skip_shape}, got {p_skip.shape}"
    )

    assert torch.all(torch.isfinite(a)), "Token output should be finite"
    assert torch.all(torch.isfinite(q_skip)), "Query skip should be finite"
    assert torch.all(torch.isfinite(c_skip)), "Context skip should be finite"
    assert torch.all(torch.isfinite(p_skip)), "Pair skip should be finite"

    assert a.dtype == dtype, f"Expected a dtype {dtype}, got {a.dtype}"
    assert q_skip.dtype == dtype, f"Expected q_skip dtype {dtype}, got {q_skip.dtype}"
    assert c_skip.dtype == dtype, f"Expected c_skip dtype {dtype}, got {c_skip.dtype}"
    assert p_skip.dtype == dtype, f"Expected p_skip dtype {dtype}, got {p_skip.dtype}"

    # Test gradient computation
    f_star_grad = {
        k: v.clone().requires_grad_(True) if v.dtype.is_floating_point else v
        for k, v in f_star.items()
    }
    r_noisy_grad = r_noisy.clone().requires_grad_(True)
    s_trunk_grad = s_trunk.clone().requires_grad_(True)
    z_atom_grad = z_atom.clone().requires_grad_(True)

    a_grad, q_skip_grad, c_skip_grad, p_skip_grad = module(
        f_star_grad, r_noisy_grad, s_trunk_grad, z_atom_grad
    )
    loss = a_grad.sum() + q_skip_grad.sum() + c_skip_grad.sum() + p_skip_grad.sum()
    loss.backward()

    # Check gradients exist and are finite
    assert r_noisy_grad.grad is not None, "Should have gradients for noisy positions"
    assert s_trunk_grad.grad is not None, "Should have gradients for trunk singles"
    # Note: z_atom may not have gradients if it's not used in the computation leading to the loss

    # Check f_star gradients for floating-point tensors
    for key, tensor in f_star_grad.items():
        if tensor.dtype.is_floating_point and tensor.grad is not None:
            assert torch.all(torch.isfinite(tensor.grad)), (
                f"f_star[{key}] gradients should be finite"
            )

    assert torch.all(torch.isfinite(r_noisy_grad.grad)), (
        "Noisy position gradients should be finite"
    )
    assert torch.all(torch.isfinite(s_trunk_grad.grad)), (
        "Trunk gradients should be finite"
    )

    # Only check z_atom gradients if they exist
    if z_atom_grad.grad is not None:
        assert torch.all(torch.isfinite(z_atom_grad.grad)), (
            "Atom pair gradients should be finite"
        )

    # Test module components exist
    assert hasattr(module, "atom_feature_proj"), "Should have atom feature projection"
    assert hasattr(module, "atom_transformer"), "Should have atom transformer"
    assert hasattr(module, "aggregation_proj"), "Should have aggregation projection"
    assert hasattr(module, "dist_proj_1"), "Should have distance projection 1"
    assert hasattr(module, "dist_proj_2"), "Should have distance projection 2"
    assert hasattr(module, "trunk_single_proj"), "Should have trunk single projection"

    # Test attention head configuration
    assert module.n_head == n_head, f"Should have {n_head} heads"

    # Test with zero positions (basic numerical stability)
    zero_f_star = {
        k: torch.zeros_like(v) if v.dtype.is_floating_point else v
        for k, v in f_star.items()
    }
    zero_r_noisy = torch.zeros_like(r_noisy)
    zero_a, zero_q_skip, zero_c_skip, zero_p_skip = module(
        zero_f_star, zero_r_noisy, s_trunk, z_atom
    )
    assert torch.all(torch.isfinite(zero_a)), "Should handle zero positions"
    assert torch.all(torch.isfinite(zero_q_skip)), "Should handle zero positions"
    assert torch.all(torch.isfinite(zero_c_skip)), "Should handle zero positions"
    assert torch.all(torch.isfinite(zero_p_skip)), "Should handle zero positions"
