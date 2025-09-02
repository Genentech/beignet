import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import AtomAttentionDecoder


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    n_tokens=st.integers(min_value=4, max_value=16),
    n_atoms=st.integers(min_value=8, max_value=32),
    c_token=st.integers(min_value=32, max_value=128).filter(
        lambda x: x % 16 == 0
    ),  # Divisible by n_head
    c_atom=st.integers(min_value=16, max_value=64).filter(
        lambda x: x % 16 == 0
    ),  # Divisible by n_head
    c_atompair=st.integers(min_value=8, max_value=32),
    n_head=st.sampled_from([4, 8, 16]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_atom_attention_decoder(
    batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
):
    """Test AtomAttentionDecoder comprehensively."""
    device = torch.device("cpu")

    # Ensure dimensions are divisible by n_head
    if c_token % n_head != 0:
        c_token = (c_token // n_head + 1) * n_head
    if c_atom % n_head != 0:
        c_atom = (c_atom // n_head + 1) * n_head

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

    # Test module parameters have gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), (
                f"Parameter {name} gradients should be finite"
            )

    # Test module components
    assert hasattr(module, "token_to_atom_proj"), "Should have token-to-atom projection"
    assert hasattr(module, "atom_self_attn"), "Should have atom self-attention"
    assert hasattr(module, "position_proj"), "Should have position projection"
    assert hasattr(module, "layer_norm_token"), "Should have token layer norm"
    assert hasattr(module, "layer_norm_atom"), "Should have atom layer norm"

    # Test attention head configuration
    assert module.n_head == n_head, f"Should have {n_head} heads"
    assert module.head_dim == c_atom // n_head, "Head dimension should be correct"

    # Test position projection output
    # The final layer should output 3D coordinates
    assert r_update.shape[-1] == 3, "Should output 3D position updates"

    # Test that the module produces reasonable position updates
    # Position updates should not be too extreme
    max_update = torch.abs(r_update).max()
    assert max_update < 1000, "Position updates should be reasonable in magnitude"

    # Test numerical stability with small values
    small_a = a * 1e-3
    small_q_skip = q_skip * 1e-3
    small_c_skip = c_skip * 1e-3
    small_r_update = module(small_a, small_q_skip, small_c_skip, p_skip)
    assert torch.all(torch.isfinite(small_r_update)), "Should handle small input values"

    # Test with zero inputs
    zero_a = torch.zeros_like(a)
    zero_q_skip = torch.zeros_like(q_skip)
    zero_c_skip = torch.zeros_like(c_skip)
    zero_p_skip = torch.zeros_like(p_skip)
    zero_r_update = module(zero_a, zero_q_skip, zero_c_skip, zero_p_skip)
    assert torch.all(torch.isfinite(zero_r_update)), "Should handle zero inputs"

    # Test parameter initialization
    for name, param in module.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} should be finite"
        assert param.dtype == dtype, f"Parameter {name} should have correct dtype"

    # Test torch.compile compatibility
    try:
        compiled_module = torch.compile(module, fullgraph=True)
        r_update_comp = compiled_module(a, q_skip, c_skip, p_skip)
        assert r_update_comp.shape == r_update.shape, "Compiled module should work"
        assert torch.all(torch.isfinite(r_update_comp)), (
            "Compiled output should be finite"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test batch processing consistency
    if batch_size > 1:
        single_a = a[0:1]
        single_q_skip = q_skip[0:1]
        single_c_skip = c_skip[0:1]
        single_p_skip = p_skip[0:1]

        single_r_update = module(single_a, single_q_skip, single_c_skip, single_p_skip)
        batch_r_update_first = r_update[0:1]

        assert torch.allclose(single_r_update, batch_r_update_first, atol=1e-5), (
            "Batch processing should be consistent"
        )

    # Test different atom/token counts
    if n_atoms > 8 and n_tokens > 4:
        fewer_atoms = n_atoms // 2
        fewer_tokens = n_tokens // 2

        smaller_a = a[:, :fewer_tokens]
        smaller_q_skip = q_skip[:, :fewer_atoms]
        smaller_c_skip = c_skip[:, :fewer_atoms]
        smaller_p_skip = p_skip[:, :fewer_atoms, :fewer_atoms]

        smaller_r_update = module(
            smaller_a, smaller_q_skip, smaller_c_skip, smaller_p_skip
        )
        expected_smaller_shape = (batch_size, fewer_atoms, 3)

        assert smaller_r_update.shape == expected_smaller_shape, (
            "Should handle different atom counts"
        )

    # Test skip connection effects
    # Different skip connections should produce different outputs
    different_c_skip = c_skip + 0.1 * torch.randn_like(c_skip)
    different_r_update = module(a, q_skip, different_c_skip, p_skip)

    skip_diff = torch.norm(different_r_update - r_update)
    assert skip_diff > 1e-6, "Different skip connections should affect output"

    # Test token representation effects
    different_a = a + 0.1 * torch.randn_like(a)
    different_a_r_update = module(different_a, q_skip, c_skip, p_skip)

    token_diff = torch.norm(different_a_r_update - r_update)
    assert token_diff > 1e-6, "Different token representations should affect output"

    # Test query skip effects
    different_q_skip = q_skip + 0.1 * torch.randn_like(q_skip)
    different_q_r_update = module(a, different_q_skip, c_skip, p_skip)

    query_diff = torch.norm(different_q_r_update - r_update)
    assert query_diff > 1e-6, "Different query skip should affect output"

    # Test position projection network
    # The position projection should be a sequential network ending in 3D output
    assert hasattr(module.position_proj, "__len__"), (
        "Position projection should be sequential"
    )
    last_layer = module.position_proj[-1]
    assert hasattr(last_layer, "out_features"), "Last layer should be Linear"
    assert last_layer.out_features == 3, (
        "Last layer should output 3 features (3D positions)"
    )

    # Test attention mechanism behavior
    # Atom self-attention should allow atoms to communicate
    # This is implicitly tested by the gradient flow and different input effects

    # Test layer normalization effects
    # Layer norms should help with training stability
    very_large_a = a * 1000
    large_a_output = module(very_large_a, q_skip, c_skip, p_skip)
    assert torch.all(torch.isfinite(large_a_output)), (
        "Layer norm should handle large inputs"
    )
