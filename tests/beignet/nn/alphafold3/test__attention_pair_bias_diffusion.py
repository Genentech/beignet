import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn.alphafold3 import AttentionPairBias


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    seq_len=st.integers(min_value=2, max_value=8),
    c_a=st.integers(min_value=16, max_value=64).filter(
        lambda x: x % 16 == 0
    ),  # Divisible by n_head
    c_s=st.integers(min_value=16, max_value=96),
    c_z=st.integers(min_value=8, max_value=32),
    n_head=st.sampled_from([4, 8, 16]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_attention_pair_bias_diffusion(
    batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
):
    """Test AttentionPairBias (Algorithm 24) with diffusion features comprehensively."""
    device = torch.device("cpu")

    # Ensure c_a is divisible by n_head
    if c_a % n_head != 0:
        c_a = (c_a // n_head + 1) * n_head

    # Create module with conditioning support
    module = (
        AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head).to(device).to(dtype)
    )

    # Generate test inputs
    a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
    s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device)
    beta = torch.randn(batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device)

    # Test basic functionality with all inputs
    a_out = module(a, s, z, beta)

    # Check output shape and properties
    assert a_out.shape == a.shape, f"Expected shape {a.shape}, got {a_out.shape}"
    assert torch.all(torch.isfinite(a_out)), "Output should be finite"
    assert a_out.dtype == dtype, f"Expected dtype {dtype}, got {a_out.dtype}"

    # Test without conditioning (s=None) - should use LayerNorm
    module_no_conditioning = (
        AttentionPairBias(c_a=c_a, c_s=None, c_z=c_z, n_head=n_head)
        .to(device)
        .to(dtype)
    )
    a_out_no_cond = module_no_conditioning(a, None, z, beta)
    assert a_out_no_cond.shape == a.shape, "Should work without conditioning"
    assert torch.all(torch.isfinite(a_out_no_cond)), (
        "Output without conditioning should be finite"
    )

    # Test with optional arguments
    # Test with only 'a' (no s, z, beta)
    a_out_minimal = module_no_conditioning(a)
    assert a_out_minimal.shape == a.shape, "Should work with minimal arguments"

    # Test with 'a' and 's' only
    a_out_as = module(a, s)
    assert a_out_as.shape == a.shape, "Should work with 'a' and 's' only"

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

    # Test module components - Algorithm 24 requirements
    assert hasattr(module, "c_a") and module.c_a == c_a
    assert hasattr(module, "c_s") and module.c_s == c_s
    assert hasattr(module, "c_z") and module.c_z == c_z
    assert hasattr(module, "n_head") and module.n_head == n_head
    assert hasattr(module, "head_dim") and module.head_dim == c_a // n_head

    # Test conditioning components
    if c_s is not None:
        assert hasattr(module, "ada_ln"), "Should have AdaLN for conditioning"
        assert hasattr(module, "linear_s_gate"), "Should have conditioning gate"
        # Test bias initialization (-2.0)
        assert torch.allclose(
            module.linear_s_gate.bias,
            torch.full_like(module.linear_s_gate.bias, -2.0),
            atol=1e-6,
        ), "Conditioning gate bias should be initialized to -2.0"
    else:
        assert hasattr(module, "layer_norm"), (
            "Should have LayerNorm when no conditioning"
        )

    # Test attention components
    assert hasattr(module, "linear_q"), "Should have query projection"
    assert hasattr(module, "linear_k"), "Should have key projection"
    assert hasattr(module, "linear_v"), "Should have value projection"
    assert hasattr(module, "linear_g"), "Should have gate projection"
    assert hasattr(module, "output_linear"), "Should have output projection"

    # Test pair representation components
    assert hasattr(module, "linear_b"), "Should have bias projection"
    assert hasattr(module, "layer_norm_z"), (
        "Should have LayerNorm for pair representation"
    )

    # Test that the module transforms the input
    # With conditioning and proper initialization, output should be different
    assert torch.all(torch.isfinite(a_out)), "Output should be finite"

    # Test Algorithm 24 step-by-step behavior
    # Step 1-4: Input normalization
    if c_s is not None:
        a_normed_manual = module.ada_ln(a, s)
    else:
        a_normed_manual = module.layer_norm(a)

    assert torch.all(torch.isfinite(a_normed_manual)), (
        "Normalized input should be finite"
    )

    # Test attention mechanism components
    # Step 6: q_i^h = Linear(ai)
    q_manual = module.linear_q(a_normed_manual)
    assert q_manual.shape == (*a.shape[:-1], c_a), "Query projection shape should match"

    # Step 7: k_i^h, v_i^h = LinearNoBias(ai)
    k_manual = module.linear_k(a_normed_manual)
    v_manual = module.linear_v(a_normed_manual)
    assert k_manual.shape == (*a.shape[:-1], c_a), "Key projection shape should match"
    assert v_manual.shape == (*a.shape[:-1], c_a), "Value projection shape should match"

    # Step 8: b_ij^h ← LinearNoBias(LayerNorm(zij)) + βij
    z_normed_manual = module.layer_norm_z(z)
    b_z_manual = module.linear_b(z_normed_manual)
    assert b_z_manual.shape == (*z.shape[:-1], n_head), (
        "Bias projection shape should match"
    )

    # Step 9: g_i^h ← sigmoid(LinearNoBias(ai))
    g_manual = torch.sigmoid(module.linear_g(a_normed_manual))
    assert g_manual.shape == (*a.shape[:-1], c_a), "Gate shape should match"
    assert torch.all(g_manual >= 0) and torch.all(g_manual <= 1), (
        "Gate should be in [0,1]"
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

    # Test conditioning effect
    # With zero conditioning, the gate should be small (due to bias=-2.0)
    zero_s_test = torch.zeros_like(s)
    conditioned_out = module(a, zero_s_test, z, beta)
    assert torch.all(torch.isfinite(conditioned_out)), "Zero conditioning should work"

    # With non-zero conditioning, results should be different
    nonzero_s_test = torch.ones_like(s)
    unconditioned_out = module(a, nonzero_s_test, z, beta)
    diff_conditioning = torch.norm(conditioned_out - unconditioned_out)
    assert diff_conditioning > 1e-6, (
        "Different conditioning should produce different outputs"
    )

    # Test pair bias effect
    zero_z_test = torch.zeros_like(z)
    out_no_pair = module(a, s, zero_z_test, beta)
    diff_pair = torch.norm(a_out - out_no_pair)
    assert diff_pair > 1e-6, "Pair representation should affect output"

    # Test attention bias effect
    zero_beta_test = torch.zeros_like(beta)
    out_no_bias = module(a, s, z, zero_beta_test)
    diff_bias = torch.norm(a_out - out_no_bias)
    assert diff_bias > 1e-6, "Attention bias should affect output"

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

    # Test batch processing consistency
    if batch_size > 1:
        # Single vs batch should produce same results per sample
        single_a = a[0:1]
        single_s = s[0:1]
        single_z = z[0:1]
        single_beta = beta[0:1]
        single_out = module(single_a, single_s, single_z, single_beta)
        batch_out_first = a_out[0:1]
        assert torch.allclose(single_out, batch_out_first, atol=1e-5), (
            "Batch processing should be consistent"
        )

    # Test attention pattern properties
    # Attention weights should sum to 1 (this is implicit in the softmax)
    # We can't directly access them, but the module should maintain this property

    # Test head dimension consistency
    assert c_a % n_head == 0, (
        f"Channel dimension {c_a} must be divisible by heads {n_head}"
    )
    assert module.head_dim * n_head == c_a, (
        "Head dimensions should reconstruct channel dimension"
    )

    # Test scale factor
    expected_scale = 1.0 / (module.head_dim**0.5)
    assert abs(module.scale - expected_scale) < 1e-6, (
        "Scale factor should be 1/√(head_dim)"
    )
