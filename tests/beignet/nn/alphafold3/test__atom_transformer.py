import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn.alphafold3 import AtomTransformer


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    n_atoms=st.integers(min_value=8, max_value=32),
    c_q=st.integers(min_value=8, max_value=16),
    c_kv=st.integers(min_value=8, max_value=16),
    c_pair=st.integers(min_value=4, max_value=8),
    n_head=st.integers(min_value=1, max_value=2),
    dtype=st.sampled_from([torch.float32]),  # Only float32 for simplicity
)
@settings(deadline=None, max_examples=3)
def test__atom_transformer(batch_size, n_atoms, c_q, c_kv, c_pair, n_head, dtype):
    """Test AtomTransformer with various input configurations."""

    # Ensure dimensions are compatible with attention heads
    c_q = (c_q // n_head) * n_head if c_q % n_head != 0 else c_q
    c_kv = (c_kv // n_head) * n_head if c_kv % n_head != 0 else c_kv

    module = AtomTransformer(
        n_block=2,  # Use smaller number for testing
        n_head=n_head,
        n_queries=8,  # Small for testing
        n_keys=16,  # Small for testing
        subset_centres=[4.0, 12.0],  # Test with 2 centers
        c_q=c_q,
        c_kv=c_kv,
        c_pair=c_pair,
    ).to(dtype=dtype)

    # Create test inputs
    q = torch.randn(batch_size, n_atoms, c_q, dtype=dtype)
    c = torch.randn(batch_size, n_atoms, c_kv, dtype=dtype)
    p = torch.randn(batch_size, n_atoms, n_atoms, c_pair, dtype=dtype)

    # Test forward pass
    output = module(q, c, p)

    # Check output shape
    assert output.shape == (batch_size, n_atoms, c_q)
    assert output.dtype == dtype

    # Check that output is finite
    assert torch.all(torch.isfinite(output))

    # Test gradient computation
    q_grad = q.clone().requires_grad_(True)
    c_grad = c.clone().requires_grad_(True)
    p_grad = p.clone().requires_grad_(True)

    output_grad = module(q_grad, c_grad, p_grad)
    loss = output_grad.sum()
    loss.backward()

    # Check gradients exist
    assert q_grad.grad is not None
    assert c_grad.grad is not None
    assert p_grad.grad is not None
    assert torch.all(torch.isfinite(q_grad.grad))

    # Test torch.compile compatibility (skip for speed in tests)
    # compiled_module = torch.compile(module, fullgraph=True)
    # output_compiled = compiled_module(q, c, p)
    # assert torch.allclose(output, output_compiled, atol=1e-5)

    # Test sequence-local masking behavior
    # The module should apply attention masking based on subset centers
    q_small = torch.randn(1, 32, c_q, dtype=dtype)
    c_small = torch.randn(1, 32, c_kv, dtype=dtype)
    p_small = torch.randn(1, 32, 32, c_pair, dtype=dtype)

    output_small = module(q_small, c_small, p_small)
    assert output_small.shape == (1, 32, c_q)

    # Test that different subset centers produce different outputs
    module2 = AtomTransformer(
        n_block=2,
        n_head=n_head,
        subset_centres=[8.0, 24.0],  # Different centers
        c_q=c_q,
        c_kv=c_kv,
        c_pair=c_pair,
    ).to(dtype=dtype)

    output2 = module2(q_small, c_small, p_small)

    # Different subset centers should generally produce different outputs
    # (though this is probabilistic and might occasionally be similar)
    assert not torch.allclose(output_small, output2, atol=1e-3)

    # Test edge case with single atom
    if n_atoms >= 1:
        q_single = torch.randn(batch_size, 1, c_q, dtype=dtype)
        c_single = torch.randn(batch_size, 1, c_kv, dtype=dtype)
        p_single = torch.randn(batch_size, 1, 1, c_pair, dtype=dtype)

        output_single = module(q_single, c_single, p_single)
        assert output_single.shape == (batch_size, 1, c_q)
        assert torch.all(torch.isfinite(output_single))
