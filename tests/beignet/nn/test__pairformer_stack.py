import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import AttentionPairBias, PairformerStack, _PairformerStackBlock


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=8),
    c_s=st.integers(min_value=64, max_value=128).filter(lambda x: x % 16 == 0),
    c_z=st.integers(min_value=32, max_value=64).filter(lambda x: x % 4 == 0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_attention_pair_bias(batch_size, seq_len, c_s, c_z, dtype):
    """Test AttentionPairBias module comprehensively."""
    device = torch.device("cpu")

    # Create module with Algorithm 17 specifications (n_head=16)
    module = AttentionPairBias(c_s=c_s, c_z=c_z, n_head=16).to(device).to(dtype)

    # Generate test inputs
    s_i = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device)

    # Test basic functionality
    s_out = module(s_i, z_ij)

    # Check output shape and properties
    assert s_out.shape == s_i.shape, f"Expected shape {s_i.shape}, got {s_out.shape}"
    assert torch.all(torch.isfinite(s_out)), "Output should be finite"
    assert s_out.dtype == dtype, f"Expected dtype {dtype}, got {s_out.dtype}"

    # Test gradient computation
    s_grad = s_i.clone().requires_grad_(True)
    z_grad = z_ij.clone().requires_grad_(True)
    s_out_grad = module(s_grad, z_grad)
    loss = s_out_grad.sum()
    loss.backward()

    assert s_grad.grad is not None, "Should have gradients for single input"
    assert z_grad.grad is not None, "Should have gradients for pair input"
    assert torch.all(torch.isfinite(s_grad.grad)), "Single gradients should be finite"
    assert torch.all(torch.isfinite(z_grad.grad)), "Pair gradients should be finite"

    # Test module parameters have gradients
    for param in module.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), (
                "Parameter gradients should be finite"
            )

    # Test input validation
    try:
        # Mismatched batch dimensions
        wrong_z = torch.randn(
            batch_size + 1, seq_len, seq_len, c_z, dtype=dtype, device=device
        )
        module(s_i, wrong_z)
        raise AssertionError("Should raise error for mismatched batch dimensions")
    except ValueError:
        pass  # Expected

    try:
        # Wrong pair shape (sequence length mismatch)
        wrong_z = torch.randn(
            batch_size, seq_len - 1, seq_len, c_z, dtype=dtype, device=device
        )
        module(s_i, wrong_z)
        raise AssertionError("Should raise error for wrong pair shape")
    except ValueError:
        pass  # Expected

    # Test that module transforms the input
    diff = torch.norm(s_out - s_i)
    assert diff > 1e-6, "Module should transform the input"

    # Test with zero inputs
    zero_s = torch.zeros_like(s_i)
    zero_z = torch.zeros_like(z_ij)
    zero_out = module(zero_s, zero_z)
    assert torch.all(torch.isfinite(zero_out)), "Should handle zero inputs"

    # Test numerical stability with small values
    small_s = s_i * 1e-3
    small_z = z_ij * 1e-3
    small_out = module(small_s, small_z)
    assert torch.all(torch.isfinite(small_out)), "Should handle small input values"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=6),
    c_s=st.integers(min_value=64, max_value=96).filter(lambda x: x % 16 == 0),
    c_z=st.integers(min_value=32, max_value=48).filter(lambda x: x % 4 == 0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_pairformer_stack_block(batch_size, seq_len, c_s, c_z, dtype):
    """Test PairformerStackBlock module comprehensively."""
    device = torch.device("cpu")

    # Create module with Algorithm 17 specifications
    module = (
        _PairformerStackBlock(
            c_s=c_s, c_z=c_z, n_head_single=16, n_head_pair=4, dropout_rate=0.25
        )
        .to(device)
        .to(dtype)
    )

    # Generate test inputs
    s_i = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device)

    # Test basic functionality
    s_out, z_out = module(s_i, z_ij)

    # Check output shapes and properties
    assert s_out.shape == s_i.shape, (
        f"Single rep: expected {s_i.shape}, got {s_out.shape}"
    )
    assert z_out.shape == z_ij.shape, (
        f"Pair rep: expected {z_ij.shape}, got {z_out.shape}"
    )
    assert torch.all(torch.isfinite(s_out)), "Single output should be finite"
    assert torch.all(torch.isfinite(z_out)), "Pair output should be finite"
    assert s_out.dtype == dtype, f"Single output: expected {dtype}, got {s_out.dtype}"
    assert z_out.dtype == dtype, f"Pair output: expected {dtype}, got {z_out.dtype}"

    # Test gradient computation
    s_grad = s_i.clone().requires_grad_(True)
    z_grad = z_ij.clone().requires_grad_(True)
    s_out_grad, z_out_grad = module(s_grad, z_grad)
    loss = s_out_grad.sum() + z_out_grad.sum()
    loss.backward()

    assert s_grad.grad is not None, "Should have gradients for single input"
    assert z_grad.grad is not None, "Should have gradients for pair input"
    assert torch.all(torch.isfinite(s_grad.grad)), "Single gradients should be finite"
    assert torch.all(torch.isfinite(z_grad.grad)), "Pair gradients should be finite"

    # Test that module processes representations (outputs should be different)
    single_diff = torch.norm(s_out - s_i)
    pair_diff = torch.norm(z_out - z_ij)
    assert single_diff > 1e-6, "Module should transform single representation"
    assert pair_diff > 1e-6, "Module should transform pair representation"

    # Test module in eval mode vs train mode
    module.eval()
    s_eval, z_eval = module(s_i, z_ij)
    module.train()
    s_train, z_train = module(s_i, z_ij)

    # Outputs may differ due to dropout, but should be finite
    assert torch.all(torch.isfinite(s_eval)), "Eval single output should be finite"
    assert torch.all(torch.isfinite(z_eval)), "Eval pair output should be finite"
    assert torch.all(torch.isfinite(s_train)), "Train single output should be finite"
    assert torch.all(torch.isfinite(z_train)), "Train pair output should be finite"

    # Test component accessibility (Algorithm 17 step verification)
    assert hasattr(module, "triangle_mult_outgoing"), "Should have step 2 component"
    assert hasattr(module, "triangle_mult_incoming"), "Should have step 3 component"
    assert hasattr(module, "triangle_attention_starting"), (
        "Should have step 4 component"
    )
    assert hasattr(module, "triangle_attention_ending"), "Should have step 5 component"
    assert hasattr(module, "pair_transition"), "Should have step 6 component"
    assert hasattr(module, "attention_pair_bias"), "Should have step 7 component"
    assert hasattr(module, "single_transition"), "Should have step 8 component"

    # Test dropout rate is correct
    assert module.dropout_rate == 0.25, "Dropout rate should match Algorithm 17"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=5),
    n_block=st.integers(min_value=1, max_value=3),
    c_s=st.sampled_from([64, 80, 96]),  # Valid values divisible by 16
    c_z=st.sampled_from([32, 36, 40, 44, 48]),  # Valid values divisible by 4
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=3)
def test_pairformer_stack(batch_size, seq_len, n_block, c_s, c_z, dtype):
    """Test PairformerStack module comprehensively."""
    device = torch.device("cpu")

    # Create module following Algorithm 17 specifications
    module = (
        PairformerStack(
            n_block=n_block,
            c_s=c_s,
            c_z=c_z,
            n_head_single=16,  # As per Algorithm 17
            n_head_pair=4,
            dropout_rate=0.25,  # As per Algorithm 17
        )
        .to(device)
        .to(dtype)
    )

    # Generate test inputs
    s_i = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device)

    # Test basic functionality
    s_out, z_out = module(s_i, z_ij)

    # Check output shapes and properties
    expected_s_shape = (batch_size, seq_len, c_s)
    expected_z_shape = (batch_size, seq_len, seq_len, c_z)
    assert s_out.shape == expected_s_shape, (
        f"Single rep: expected {expected_s_shape}, got {s_out.shape}"
    )
    assert z_out.shape == expected_z_shape, (
        f"Pair rep: expected {expected_z_shape}, got {z_out.shape}"
    )
    assert torch.all(torch.isfinite(s_out)), "Single output should be finite"
    assert torch.all(torch.isfinite(z_out)), "Pair output should be finite"
    assert s_out.dtype == dtype, f"Single output: expected {dtype}, got {s_out.dtype}"
    assert z_out.dtype == dtype, f"Pair output: expected {dtype}, got {z_out.dtype}"

    # Test gradient computation
    s_grad = s_i.clone().requires_grad_(True)
    z_grad = z_ij.clone().requires_grad_(True)
    s_out_grad, z_out_grad = module(s_grad, z_grad)
    loss = s_out_grad.sum() + z_out_grad.sum()
    loss.backward()

    assert s_grad.grad is not None, "Should have gradients for single input"
    assert z_grad.grad is not None, "Should have gradients for pair input"
    assert torch.all(torch.isfinite(s_grad.grad)), "Single gradients should be finite"
    assert torch.all(torch.isfinite(z_grad.grad)), "Pair gradients should be finite"

    # Test module parameters have gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), (
                f"Parameter {name} gradients should be finite"
            )

    # Test input validation
    try:
        # Mismatched sequence lengths
        wrong_s = s_i[:, : seq_len - 1] if seq_len > 1 else s_i
        module(wrong_s, z_ij)
        if seq_len > 1:
            raise AssertionError("Should raise error for mismatched sequence lengths")
    except ValueError:
        pass  # Expected

    try:
        # Wrong single channel dimension
        wrong_s = torch.randn(batch_size, seq_len, c_s + 1, dtype=dtype, device=device)
        module(wrong_s, z_ij)
        raise AssertionError("Should raise error for wrong single channel dimension")
    except ValueError:
        pass  # Expected

    try:
        # Wrong pair channel dimension
        wrong_z = torch.randn(
            batch_size, seq_len, seq_len, c_z + 1, dtype=dtype, device=device
        )
        module(s_i, wrong_z)
        raise AssertionError("Should raise error for wrong pair channel dimension")
    except ValueError:
        pass  # Expected

    # Test that PairformerStack processes representations through all blocks
    single_diff = torch.norm(s_out - s_i)
    pair_diff = torch.norm(z_out - z_ij)
    assert single_diff > 1e-6, "PairformerStack should transform single representation"
    assert pair_diff > 1e-6, "PairformerStack should transform pair representation"

    # Test module state consistency
    module.eval()
    s_eval, z_eval = module(s_i, z_ij)
    module.train()

    # In eval mode, outputs may differ due to dropout but should be finite
    assert torch.all(torch.isfinite(s_eval)), "Eval single output should be finite"
    assert torch.all(torch.isfinite(z_eval)), "Eval pair output should be finite"

    # Test with different block counts
    if n_block > 1:
        single_block_module = (
            PairformerStack(
                n_block=1, c_s=c_s, c_z=c_z, n_head_single=16, n_head_pair=4
            )
            .to(device)
            .to(dtype)
        )

        single_s_out, single_z_out = single_block_module(s_i, z_ij)
        assert single_s_out.shape == expected_s_shape, (
            "Single block should produce correct shape"
        )
        assert single_z_out.shape == expected_z_shape, (
            "Single block should produce correct shape"
        )
        assert torch.all(torch.isfinite(single_s_out)), (
            "Single block single output should be finite"
        )
        assert torch.all(torch.isfinite(single_z_out)), (
            "Single block pair output should be finite"
        )

    # Test parameter count is reasonable
    total_params = sum(p.numel() for p in module.parameters())
    assert total_params > 0, "Module should have parameters"

    # With multiple blocks, should have more parameters
    if n_block > 1:
        assert total_params > c_s * c_z, (
            "Multi-block module should have reasonable parameter count"
        )

    # Test component accessibility and Algorithm 17 compliance
    assert len(module.blocks) == n_block, f"Should have {n_block} blocks"
    for i, block in enumerate(module.blocks):
        assert isinstance(block, _PairformerStackBlock), (
            f"Block {i} should be PairformerStackBlock"
        )

        # Verify each block has all Algorithm 17 components
        assert hasattr(block, "triangle_mult_outgoing"), (
            f"Block {i} should have step 2 component"
        )
        assert hasattr(block, "triangle_mult_incoming"), (
            f"Block {i} should have step 3 component"
        )
        assert hasattr(block, "triangle_attention_starting"), (
            f"Block {i} should have step 4 component"
        )
        assert hasattr(block, "triangle_attention_ending"), (
            f"Block {i} should have step 5 component"
        )
        assert hasattr(block, "pair_transition"), (
            f"Block {i} should have step 6 component"
        )
        assert hasattr(block, "attention_pair_bias"), (
            f"Block {i} should have step 7 component"
        )
        assert hasattr(block, "single_transition"), (
            f"Block {i} should have step 8 component"
        )

    # Test numerical stability with small values
    small_s = s_i * 1e-3
    small_z = z_ij * 1e-3
    small_s_out, small_z_out = module(small_s, small_z)
    assert torch.all(torch.isfinite(small_s_out)), (
        "Should handle small single input values"
    )
    assert torch.all(torch.isfinite(small_z_out)), (
        "Should handle small pair input values"
    )

    # Test with zero inputs
    zero_s = torch.zeros_like(s_i)
    zero_z = torch.zeros_like(z_ij)
    zero_s_out, zero_z_out = module(zero_s, zero_z)
    assert torch.all(torch.isfinite(zero_s_out)), "Should handle zero single inputs"
    assert torch.all(torch.isfinite(zero_z_out)), "Should handle zero pair inputs"

    # Test that all parameters are properly initialized
    for name, param in module.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} should be finite"
        assert param.dtype == dtype, f"Parameter {name} should have correct dtype"

    # Verify Algorithm 17 specifications are met
    assert module.n_block == n_block, "n_block should match Algorithm 17 N_block"

    # Check that the first block has correct specifications
    first_block = module.blocks[0]
    assert hasattr(first_block, "attention_pair_bias"), "Should have AttentionPairBias"
    assert first_block.attention_pair_bias.n_head == 16, (
        "Should use 16 heads for single attention (Algorithm 17)"
    )
    assert first_block.dropout_rate == 0.25, (
        "Should use 0.25 dropout rate (Algorithm 17)"
    )
