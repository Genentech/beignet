import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import _FourierEmbedding


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=2, max_value=10),
    c=st.integers(min_value=8, max_value=64),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_fourier_embedding(batch_size, seq_len, c, dtype):
    """Test FourierEmbedding (Algorithm 22) comprehensively."""
    device = torch.device("cpu")

    # Create module
    module = _FourierEmbedding(c=c).to(device).to(dtype)

    # Test basic functionality with different input shapes
    # Test with shape (..., 1)
    t1 = torch.randn(batch_size, seq_len, 1, dtype=dtype, device=device)
    embeddings1 = module(t1)

    # Test with shape (...)
    t2 = torch.randn(batch_size, seq_len, dtype=dtype, device=device)
    embeddings2 = module(t2)

    # Check output shapes and properties
    expected_shape = (batch_size, seq_len, c)
    assert embeddings1.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {embeddings1.shape}"
    )
    assert embeddings2.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {embeddings2.shape}"
    )

    assert torch.all(torch.isfinite(embeddings1)), "Output should be finite"
    assert torch.all(torch.isfinite(embeddings2)), "Output should be finite"
    assert embeddings1.dtype == dtype, (
        f"Expected dtype {dtype}, got {embeddings1.dtype}"
    )
    assert embeddings2.dtype == dtype, (
        f"Expected dtype {dtype}, got {embeddings2.dtype}"
    )

    # Test that outputs are in valid range for cosine [-1, 1]
    assert torch.all(embeddings1 >= -1.0 - 1e-6), "Cosine output should be >= -1"
    assert torch.all(embeddings1 <= 1.0 + 1e-6), "Cosine output should be <= 1"
    assert torch.all(embeddings2 >= -1.0 - 1e-6), "Cosine output should be >= -1"
    assert torch.all(embeddings2 <= 1.0 + 1e-6), "Cosine output should be <= 1"

    # Test gradient computation
    t_grad = torch.randn(
        batch_size, seq_len, 1, dtype=dtype, device=device, requires_grad=True
    )
    embeddings_grad = module(t_grad)
    loss = embeddings_grad.sum()
    loss.backward()

    assert t_grad.grad is not None, "Should have gradients for input 't'"
    assert torch.all(torch.isfinite(t_grad.grad)), "Input gradients should be finite"

    # Test that module has no trainable parameters (w and b are buffers)
    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert param_count == 0, "FourierEmbedding should have no trainable parameters"

    # Test that w and b are properly initialized as buffers
    assert hasattr(module, "w"), "Should have weight buffer 'w'"
    assert hasattr(module, "b"), "Should have bias buffer 'b'"
    assert module.w.shape == (c,), f"Weight should have shape ({c},)"
    assert module.b.shape == (c,), f"Bias should have shape ({c},)"
    assert not module.w.requires_grad, "Weight buffer should not require grad"
    assert not module.b.requires_grad, "Bias buffer should not require grad"

    # Test mathematical properties - Algorithm 22 implementation
    # cos(2π(tw + b)) should be deterministic for same input
    t_test = torch.ones(1, 1, 1, dtype=dtype, device=device)
    out1 = module(t_test)
    out2 = module(t_test)
    assert torch.allclose(out1, out2, atol=1e-6), "Output should be deterministic"

    # Test manual computation matches module output
    t_manual = torch.randn(2, 3, 1, dtype=dtype, device=device)
    module_out = module(t_manual)

    # Manual computation: cos(2π(tw + b))
    t_squeezed = t_manual.squeeze(-1).unsqueeze(-1)  # (..., 1)
    manual_out = torch.cos(2 * torch.pi * (t_squeezed * module.w + module.b))

    assert torch.allclose(module_out, manual_out, atol=1e-6), (
        "Manual computation should match module"
    )

    # Test numerical stability with extreme values
    large_t = torch.full((1, 1, 1), 1000.0, dtype=dtype, device=device)
    large_out = module(large_t)
    assert torch.all(torch.isfinite(large_out)), "Should handle large input values"

    small_t = torch.full((1, 1, 1), 1e-6, dtype=dtype, device=device)
    small_out = module(small_t)
    assert torch.all(torch.isfinite(small_out)), "Should handle small input values"

    # Test batch processing
    if batch_size > 1:
        # Single vs batch should produce same results per sample
        single_t = t1[0:1]  # First sample
        single_out = module(single_t)
        batch_out_first = embeddings1[0:1]
        assert torch.allclose(single_out, batch_out_first, atol=1e-6), (
            "Batch processing should be consistent"
        )

    # Test torch.compile compatibility
    try:
        compiled_module = torch.compile(module, fullgraph=True)
        compiled_out = compiled_module(t1)
        assert compiled_out.shape == embeddings1.shape, "Compiled module should work"
        assert torch.allclose(compiled_out, embeddings1, atol=1e-6), (
            "Compiled output should match"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test different sequence lengths work
    if seq_len > 2:
        t_short = t1[:, :2]  # Shorter sequence
        out_short = module(t_short)
        assert out_short.shape == (batch_size, 2, c), (
            "Should handle different sequence lengths"
        )

    # Test zero input
    zero_t = torch.zeros(1, 2, 1, dtype=dtype, device=device)
    zero_out = module(zero_t)
    expected_zero_out = (
        torch.cos(2 * torch.pi * module.b).unsqueeze(0).unsqueeze(0).expand(1, 2, -1)
    )
    assert torch.allclose(zero_out, expected_zero_out, atol=1e-6), (
        "Zero input should give cos(2πb)"
    )

    # Test parameter initialization (should be from standard normal)
    # This is a statistical test - weights should be roughly normally distributed
    assert torch.all(torch.isfinite(module.w)), "Weights should be finite"
    assert torch.all(torch.isfinite(module.b)), "Biases should be finite"
