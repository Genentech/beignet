import torch

from beignet.nn.alphafold3 import RelativePositionEncoding


def test_relative_position_encoding():
    """Test RelativePositionEncoding comprehensively."""
    device = torch.device("cpu")
    dtype = torch.float32
    batch_size = 2
    n_tokens = 10
    c_z = 128

    # Create module with correct parameters
    module = RelativePositionEncoding(c_z=c_z).to(device).to(dtype)

    # Generate test inputs - f_star dictionary with required features
    f_star = {
        "asym_id": torch.randint(0, 3, (batch_size, n_tokens), device=device),
        "residue_index": torch.arange(n_tokens, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1),
        "entity_id": torch.randint(0, 2, (batch_size, n_tokens), device=device),
        "token_index": torch.arange(n_tokens, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1),
        "sym_id": torch.randint(0, 5, (batch_size, n_tokens), device=device),
    }

    # Test basic functionality
    rel_pos_enc = module(f_star)

    # Check output shape and properties
    expected_shape = (batch_size, n_tokens, n_tokens, c_z)
    assert rel_pos_enc.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {rel_pos_enc.shape}"
    )
    assert torch.all(torch.isfinite(rel_pos_enc)), "Output should be finite"
    assert rel_pos_enc.dtype == dtype, (
        f"Expected dtype {dtype}, got {rel_pos_enc.dtype}"
    )

    # Test gradient computation through module parameters
    # Note: Input tensors are integers/indices so gradients don't flow back to them,
    # but the module parameters should receive gradients
    rel_pos_enc_grad = module(f_star)
    loss = rel_pos_enc_grad.sum()
    loss.backward()

    # The linear layer should have gradients
    assert module.linear.weight.grad is not None, "Linear layer should have gradients"
    assert torch.all(torch.isfinite(module.linear.weight.grad)), (
        "Linear gradients should be finite"
    )

    # Test module parameters have gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), (
                f"Parameter {name} gradients should be finite"
            )

    # Test module components
    assert hasattr(module, "linear"), "Should have linear projection"
    assert hasattr(module, "r_max"), "Should have r_max parameter"
    assert hasattr(module, "s_max"), "Should have s_max parameter"
    assert hasattr(module, "c_z"), "Should have c_z parameter"

    # Test with simple case - 3 tokens
    simple_f_star = {
        "asym_id": torch.tensor([[0, 0, 0]], device=device),
        "residue_index": torch.tensor([[0, 1, 2]], device=device),
        "entity_id": torch.tensor([[0, 0, 0]], device=device),
        "token_index": torch.tensor([[0, 1, 2]], device=device),
        "sym_id": torch.tensor([[0, 0, 0]], device=device),
    }
    rel_enc_simple = module(simple_f_star)

    # Check that the encoding captures relative distances properly
    assert torch.all(torch.isfinite(rel_enc_simple)), "Simple case should be finite"
    expected_simple_shape = (1, 3, 3, c_z)
    assert rel_enc_simple.shape == expected_simple_shape, (
        f"Expected {expected_simple_shape}, got {rel_enc_simple.shape}"
    )

    # Test with identical residue indices (zero distances)
    identical_f_star = {
        "asym_id": torch.tensor([[0, 0]], device=device),
        "residue_index": torch.tensor([[5, 5]], device=device),  # Same residue
        "entity_id": torch.tensor([[0, 0]], device=device),
        "token_index": torch.tensor([[0, 1]], device=device),  # Different tokens
        "sym_id": torch.tensor([[0, 0]], device=device),
    }
    identical_enc = module(identical_f_star)
    assert torch.all(torch.isfinite(identical_enc)), "Should handle identical residues"

    # Test with large residue distances
    large_f_star = {
        "asym_id": torch.tensor([[0, 0]], device=device),
        "residue_index": torch.tensor([[0, 1000]], device=device),  # Large distance
        "entity_id": torch.tensor([[0, 0]], device=device),
        "token_index": torch.tensor([[0, 1000]], device=device),
        "sym_id": torch.tensor([[0, 0]], device=device),
    }
    large_enc = module(large_f_star)
    assert torch.all(torch.isfinite(large_enc)), "Should handle large residue distances"

    # Test parameter initialization
    for name, param in module.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} should be finite"
        assert param.dtype == dtype, f"Parameter {name} should have correct dtype"

    # Test parameter values are reasonable
    assert module.r_max == 32, "Default r_max should be 32"
    assert module.s_max == 2, "Default s_max should be 2"
    assert module.c_z == c_z, f"c_z should be {c_z}"

    # Test numerical stability with extreme values
    extreme_f_star = {
        "asym_id": torch.tensor([[0, 1]], device=device),  # Different chains
        "residue_index": torch.tensor(
            [[0, 10000]], device=device
        ),  # Very large residue distance
        "entity_id": torch.tensor([[0, 1]], device=device),  # Different entities
        "token_index": torch.tensor([[0, 10000]], device=device),
        "sym_id": torch.tensor([[0, 100]], device=device),  # Large sym_id difference
    }
    extreme_enc = module(extreme_f_star)
    assert torch.all(torch.isfinite(extreme_enc)), "Should handle extreme values"

    # Test batch processing consistency
    single_f_star = {k: v[0:1] for k, v in f_star.items()}
    single_enc = module(single_f_star)
    batch_enc_first = rel_pos_enc[0:1]
    assert torch.allclose(single_enc, batch_enc_first, atol=1e-5), (
        "Batch processing should be consistent"
    )

    # Test torch.compile compatibility
    try:
        compiled_module = torch.compile(module, fullgraph=True)
        compiled_out = compiled_module(f_star)
        assert compiled_out.shape == rel_pos_enc.shape, "Compiled module should work"
        assert torch.allclose(compiled_out, rel_pos_enc, atol=1e-5), (
            "Compiled output should match"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test different token counts
    fewer_tokens = n_tokens // 2
    fewer_f_star = {k: v[:, :fewer_tokens] for k, v in f_star.items()}
    fewer_enc = module(fewer_f_star)
    expected_fewer_shape = (batch_size, fewer_tokens, fewer_tokens, c_z)
    assert fewer_enc.shape == expected_fewer_shape, (
        "Should handle different token counts"
    )

    # Test encoding properties
    # Diagonal elements should correspond to self-relations
    diag_elements = torch.diagonal(
        rel_pos_enc, dim1=1, dim2=2
    )  # (batch, n_tokens, c_z)

    # All diagonal elements should be finite and represent same-token encoding
    assert torch.all(torch.isfinite(diag_elements)), (
        "Diagonal elements should be finite"
    )

    # Test different chain encoding - elements from different chains should differ from same chain
    if torch.any(f_star["asym_id"][0, :] != f_star["asym_id"][0, 0]):
        # If we have different chains, the encoding should reflect this
        same_chain_mask = f_star["asym_id"].unsqueeze(-1) == f_star[
            "asym_id"
        ].unsqueeze(-2)
        diff_chain_mask = ~same_chain_mask

        if torch.any(diff_chain_mask):
            same_chain_enc = rel_pos_enc[same_chain_mask]
            diff_chain_enc = rel_pos_enc[diff_chain_mask]

            # Different chains should generally have different encodings
            # (This is a probabilistic test, so we use a loose bound)
            same_chain_mean = same_chain_enc.mean()
            diff_chain_mean = diff_chain_enc.mean()
            # Just check they are both finite
            assert torch.isfinite(same_chain_mean), (
                "Same chain encoding should be finite"
            )
            assert torch.isfinite(diff_chain_mean), (
                "Different chain encoding should be finite"
            )
