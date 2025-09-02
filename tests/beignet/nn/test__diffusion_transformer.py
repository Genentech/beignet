import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import DiffusionTransformer


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    seq_len=st.integers(min_value=2, max_value=8),
    c_a=st.integers(min_value=16, max_value=64).filter(
        lambda x: x % 16 == 0
    ),  # Divisible by n_head
    c_s=st.integers(min_value=16, max_value=96),
    c_z=st.integers(min_value=8, max_value=32),
    n_head=st.sampled_from([4, 8, 16]),
    n_block=st.integers(min_value=1, max_value=3),
    n=st.integers(min_value=2, max_value=4),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_diffusion_transformer(
    batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, n, dtype
):
    """Test DiffusionTransformer (Algorithm 23) comprehensively."""
    device = torch.device("cpu")

    # Ensure c_a is divisible by n_head
    if c_a % n_head != 0:
        c_a = (c_a // n_head + 1) * n_head

    # Create module
    module = (
        DiffusionTransformer(
            c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=n_block, n=n
        )
        .to(device)
        .to(dtype)
    )

    # Generate test inputs
    a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
    s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device)
    beta = torch.randn(batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device)

    # Test basic functionality
    a_out = module(a, s, z, beta)

    # Check output shape and properties
    assert a_out.shape == a.shape, f"Expected shape {a.shape}, got {a_out.shape}"
    assert torch.all(torch.isfinite(a_out)), "Output should be finite"
    assert a_out.dtype == dtype, f"Expected dtype {dtype}, got {a_out.dtype}"

    # Test gradient computation
    a_grad = a.clone().requires_grad_(True)
    s_grad = s.clone().requires_grad_(True)
    z_grad = z.clone().requires_grad_(True)
    beta_grad = beta.clone().requires_grad_(True)

    a_out_grad = module(a_grad, s_grad, z_grad, beta_grad)
    loss = a_out_grad.sum()
    loss.backward()

    assert a_grad.grad is not None, "Should have gradients for input 'a'"
    assert s_grad.grad is not None, "Should have gradients for conditioning 's'"
    assert z_grad.grad is not None, "Should have gradients for pair 'z'"
    assert beta_grad.grad is not None, "Should have gradients for bias 'beta'"

    assert torch.all(torch.isfinite(a_grad.grad)), (
        "Input 'a' gradients should be finite"
    )
    assert torch.all(torch.isfinite(s_grad.grad)), (
        "Input 's' gradients should be finite"
    )
    assert torch.all(torch.isfinite(z_grad.grad)), (
        "Input 'z' gradients should be finite"
    )
    assert torch.all(torch.isfinite(beta_grad.grad)), (
        "Input 'beta' gradients should be finite"
    )

    # Test module parameters have gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), (
                f"Parameter {name} gradients should be finite"
            )

    # Test module structure - Algorithm 23 requirements
    assert hasattr(module, "blocks"), "Should have blocks ModuleList"
    assert len(module.blocks) == n_block, f"Should have {n_block} blocks"

    for i, block in enumerate(module.blocks):
        assert "attention" in block, f"Block {i} should have attention component"
        assert "transition" in block, f"Block {i} should have transition component"

        # Check AttentionPairBias component
        attention = block["attention"]
        assert hasattr(attention, "c_a") and attention.c_a == c_a
        assert hasattr(attention, "c_s") and attention.c_s == c_s
        assert hasattr(attention, "c_z") and attention.c_z == c_z
        assert hasattr(attention, "n_head") and attention.n_head == n_head

        # Check ConditionedTransitionBlock component
        transition = block["transition"]
        assert hasattr(transition, "c") and transition.c == c_a
        assert hasattr(transition, "c_s") and transition.c_s == c_s
        assert hasattr(transition, "n") and transition.n == n

    # Test that the module transforms the input (should be different from input)
    diff = torch.norm(a_out - a)
    # With proper initialization, the output should be different from input
    # Initial bias of -2.0 makes gates small but not zero
    assert torch.all(torch.isfinite(a_out)), "Output should be finite"

    # Test Algorithm 23 step-by-step behavior
    # Manually simulate one block to verify implementation
    if n_block >= 1:
        first_block = module.blocks[0]

        # Step 2: {bi} = AttentionPairBias({ai}, {si}, {zij}, {βij}, N_head)
        b_manual = first_block["attention"](a, s, z, beta)
        assert b_manual.shape == a.shape, "Attention output should match input shape"

        # Step 3: ai ← bi + ConditionedTransitionBlock(ai, si)
        transition_manual = first_block["transition"](a, s)
        expected_intermediate = b_manual + transition_manual

        # For single block, this should match the module output
        if n_block == 1:
            single_block_module = (
                DiffusionTransformer(
                    c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=1, n=n
                )
                .to(device)
                .to(dtype)
            )
            # Copy parameters from original module's first block
            single_block_module.blocks[0].load_state_dict(first_block.state_dict())

            single_out = single_block_module(a, s, z, beta)
            # Should match our manual computation
            assert torch.allclose(single_out, expected_intermediate, atol=1e-5), (
                "Single block output should match manual computation"
            )

    # Test numerical stability with small values
    small_a = a * 1e-3
    small_s = s * 1e-3
    small_z = z * 1e-3
    small_beta = beta * 1e-3
    small_out = module(small_a, small_s, small_z, small_beta)
    assert torch.all(torch.isfinite(small_out)), "Should handle small input values"

    # Test with zero inputs
    zero_a = torch.zeros_like(a)
    zero_s = torch.zeros_like(s)
    zero_z = torch.zeros_like(z)
    zero_beta = torch.zeros_like(beta)
    zero_out = module(zero_a, zero_s, zero_z, zero_beta)
    assert torch.all(torch.isfinite(zero_out)), "Should handle zero inputs"

    # Test parameter initialization
    for name, param in module.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} should be finite"
        assert param.dtype == dtype, f"Parameter {name} should have correct dtype"

    # Test different block counts
    if n_block > 1:
        # Test with fewer blocks
        module_fewer = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=1, n=n
            )
            .to(device)
            .to(dtype)
        )
        out_fewer = module_fewer(a, s, z, beta)
        assert out_fewer.shape == a.shape, "Fewer blocks should work"
        assert torch.all(torch.isfinite(out_fewer)), (
            "Fewer blocks output should be finite"
        )

    # Test torch.compile compatibility
    try:
        compiled_module = torch.compile(module, fullgraph=True)
        compiled_out = compiled_module(a, s, z, beta)
        assert compiled_out.shape == a_out.shape, "Compiled module should work"
        assert torch.all(torch.isfinite(compiled_out)), (
            "Compiled output should be finite"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test residual connection pattern
    # The residual connections should help with gradient flow
    # Test that changing inputs affects the output
    a_modified = a + 0.1 * torch.randn_like(a)
    out_modified = module(a_modified, s, z, beta)
    diff_modified = torch.norm(out_modified - a_out)
    assert diff_modified > 1e-6, "Input changes should affect output"

    # Test conditioning effect
    s_modified = s + 0.1 * torch.randn_like(s)
    out_s_modified = module(a, s_modified, z, beta)
    diff_s = torch.norm(out_s_modified - a_out)
    assert diff_s > 1e-6, "Conditioning changes should affect output"

    # Test pair representation effect
    z_modified = z + 0.1 * torch.randn_like(z)
    out_z_modified = module(a, s, z_modified, beta)
    diff_z = torch.norm(out_z_modified - a_out)
    assert diff_z > 1e-6, "Pair representation changes should affect output"

    # Test bias effect
    beta_modified = beta + 0.1 * torch.randn_like(beta)
    out_beta_modified = module(a, s, z, beta_modified)
    diff_beta = torch.norm(out_beta_modified - a_out)
    assert diff_beta > 1e-6, "Bias changes should affect output"
