import torch

from beignet.nn.alphafold3 import AtomAttentionDecoder


def test_atom_attention_decoder():
    """Test AtomAttentionDecoder basic functionality."""
    # Fixed test parameters for simplicity
    batch_size = 2
    n_tokens = 8
    n_atoms = 12
    c_token = 64  # Divisible by n_head
    c_atom = 32  # Divisible by n_head
    c_atompair = 16
    n_head = 8
    dtype = torch.float32
    device = torch.device("cpu")

    # Create module
    module = (
        AtomAttentionDecoder(c_token=c_token, c_atom=c_atom, n_head=n_head)
        .to(device)
        .to(dtype)
    )

    # Generate test inputs
    a = torch.randn(batch_size, n_tokens, c_token, dtype=dtype, device=device)
    q_skip = torch.randn(batch_size, n_atoms, c_token, dtype=dtype, device=device)
    c_skip = torch.randn(batch_size, n_atoms, c_atom, dtype=dtype, device=device)
    p_skip = torch.randn(
        batch_size, n_atoms, n_atoms, c_atompair, dtype=dtype, device=device
    )

    # Test basic functionality
    r_update = module(a, q_skip, c_skip, p_skip)

    # Check output shape and properties
    expected_shape = (batch_size, n_atoms, 3)
    assert r_update.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {r_update.shape}"
    )
    assert torch.all(torch.isfinite(r_update)), "Position updates should be finite"
    assert r_update.dtype == dtype, f"Expected dtype {dtype}, got {r_update.dtype}"

    # Test gradient computation
    a_grad = a.clone().requires_grad_(True)
    q_skip_grad = q_skip.clone().requires_grad_(True)
    c_skip_grad = c_skip.clone().requires_grad_(True)
    p_skip_grad = p_skip.clone().requires_grad_(True)

    r_update_grad = module(a_grad, q_skip_grad, c_skip_grad, p_skip_grad)
    loss = r_update_grad.sum()
    loss.backward()

    # Check gradients exist and are finite
    assert a_grad.grad is not None, "Should have gradients for token representations"
    assert q_skip_grad.grad is not None, "Should have gradients for query skip"
    assert c_skip_grad.grad is not None, "Should have gradients for context skip"
    assert p_skip_grad.grad is not None, "Should have gradients for pair skip"

    assert torch.all(torch.isfinite(a_grad.grad)), "Token gradients should be finite"
    assert torch.all(torch.isfinite(q_skip_grad.grad)), (
        "Query skip gradients should be finite"
    )
    assert torch.all(torch.isfinite(c_skip_grad.grad)), (
        "Context skip gradients should be finite"
    )
    assert torch.all(torch.isfinite(p_skip_grad.grad)), (
        "Pair skip gradients should be finite"
    )

    # Test module components exist
    assert hasattr(module, "token_to_atom_proj"), "Should have token-to-atom projection"
    assert hasattr(module, "atom_transformer"), "Should have atom transformer"
    assert hasattr(module, "position_proj"), "Should have position projection"
    assert hasattr(module, "layer_norm"), "Should have layer norm"

    # Test attention head configuration
    assert module.n_head == n_head, f"Should have {n_head} heads"

    # Test position projection output dimensions
    assert r_update.shape[-1] == 3, "Should output 3D position updates"

    # Test with zero inputs (basic numerical stability)
    zero_a = torch.zeros_like(a)
    zero_q_skip = torch.zeros_like(q_skip)
    zero_c_skip = torch.zeros_like(c_skip)
    zero_p_skip = torch.zeros_like(p_skip)
    zero_r_update = module(zero_a, zero_q_skip, zero_c_skip, zero_p_skip)
    assert torch.all(torch.isfinite(zero_r_update)), "Should handle zero inputs"

    # Test position projection network structure
    assert hasattr(module.position_proj, "out_features"), (
        "Position projection should be Linear"
    )
    assert module.position_proj.out_features == 3, (
        "Position projection should output 3 features"
    )
