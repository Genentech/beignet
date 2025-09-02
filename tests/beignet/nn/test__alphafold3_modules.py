import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import (
    MSAPairWeightedAveraging,
    Transition,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=6),
    c=st.integers(min_value=8, max_value=32).filter(
        lambda x: x % 4 == 0
    ),  # Ensure divisible by n_head
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_triangle_multiplication_outgoing(batch_size, seq_len, c, dtype):
    """Test TriangleMultiplicationOutgoing module."""
    device = torch.device("cpu")

    # Create module
    module = TriangleMultiplicationOutgoing(c=c).to(device).to(dtype)

    # Generate test input
    z_ij = torch.randn(batch_size, seq_len, seq_len, c, dtype=dtype, device=device)

    # Test basic functionality
    z_out = module(z_ij)

    # Check output shape and properties
    assert z_out.shape == z_ij.shape, f"Expected shape {z_ij.shape}, got {z_out.shape}"
    assert torch.all(torch.isfinite(z_out)), "Output should be finite"
    assert z_out.dtype == dtype, f"Expected dtype {dtype}, got {z_out.dtype}"

    # Test gradient computation
    z_grad = z_ij.clone().requires_grad_(True)
    z_out_grad = module(z_grad)
    loss = z_out_grad.sum()
    loss.backward()

    assert z_grad.grad is not None, "Should have gradients for input"
    assert torch.all(torch.isfinite(z_grad.grad)), "Input gradients should be finite"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=6),
    c=st.integers(min_value=8, max_value=32).filter(lambda x: x % 4 == 0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_triangle_multiplication_incoming(batch_size, seq_len, c, dtype):
    """Test TriangleMultiplicationIncoming module."""
    device = torch.device("cpu")

    module = TriangleMultiplicationIncoming(c=c).to(device).to(dtype)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c, dtype=dtype, device=device)

    z_out = module(z_ij)

    assert z_out.shape == z_ij.shape, f"Expected shape {z_ij.shape}, got {z_out.shape}"
    assert torch.all(torch.isfinite(z_out)), "Output should be finite"
    assert z_out.dtype == dtype, f"Expected dtype {dtype}, got {z_out.dtype}"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=6),
    c=st.integers(min_value=8, max_value=32).filter(lambda x: x % 4 == 0),
    n_head=st.integers(min_value=2, max_value=4),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_triangle_attention_starting_node(batch_size, seq_len, c, n_head, dtype):
    """Test TriangleAttentionStartingNode module."""
    # Ensure c is divisible by n_head
    if c % n_head != 0:
        c = (c // n_head) * n_head

    device = torch.device("cpu")

    module = TriangleAttentionStartingNode(c=c, n_head=n_head).to(device).to(dtype)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c, dtype=dtype, device=device)

    z_out = module(z_ij)

    assert z_out.shape == z_ij.shape, f"Expected shape {z_ij.shape}, got {z_out.shape}"
    assert torch.all(torch.isfinite(z_out)), "Output should be finite"
    assert z_out.dtype == dtype, f"Expected dtype {dtype}, got {z_out.dtype}"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=6),
    c=st.integers(min_value=8, max_value=32).filter(lambda x: x % 4 == 0),
    n_head=st.integers(min_value=2, max_value=4),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_triangle_attention_ending_node(batch_size, seq_len, c, n_head, dtype):
    """Test TriangleAttentionEndingNode module."""
    # Ensure c is divisible by n_head
    if c % n_head != 0:
        c = (c // n_head) * n_head

    device = torch.device("cpu")

    module = TriangleAttentionEndingNode(c=c, n_head=n_head).to(device).to(dtype)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c, dtype=dtype, device=device)

    z_out = module(z_ij)

    assert z_out.shape == z_ij.shape, f"Expected shape {z_ij.shape}, got {z_out.shape}"
    assert torch.all(torch.isfinite(z_out)), "Output should be finite"
    assert z_out.dtype == dtype, f"Expected dtype {dtype}, got {z_out.dtype}"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=6),
    n_seq=st.integers(min_value=2, max_value=5),
    c_m=st.integers(min_value=8, max_value=32).filter(lambda x: x % 4 == 0),
    c_z=st.integers(min_value=8, max_value=32),
    n_head=st.integers(min_value=2, max_value=4),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_msa_pair_weighted_averaging(
    batch_size, seq_len, n_seq, c_m, c_z, n_head, dtype
):
    """Test MSAPairWeightedAveraging module."""
    # Ensure c_m is divisible by n_head
    if c_m % n_head != 0:
        c_m = (c_m // n_head) * n_head

    device = torch.device("cpu")

    module = (
        MSAPairWeightedAveraging(c_m=c_m, c_z=c_z, n_head=n_head).to(device).to(dtype)
    )

    m_si = torch.randn(batch_size, seq_len, n_seq, c_m, dtype=dtype, device=device)
    z_ij = torch.randn(batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device)

    m_out = module(m_si, z_ij)

    assert m_out.shape == m_si.shape, f"Expected shape {m_si.shape}, got {m_out.shape}"
    assert torch.all(torch.isfinite(m_out)), "Output should be finite"
    assert m_out.dtype == dtype, f"Expected dtype {dtype}, got {m_out.dtype}"

    # Test gradient computation
    m_grad = m_si.clone().requires_grad_(True)
    z_grad = z_ij.clone().requires_grad_(True)
    m_out_grad = module(m_grad, z_grad)
    loss = m_out_grad.sum()
    loss.backward()

    assert m_grad.grad is not None, "Should have gradients for MSA input"
    assert z_grad.grad is not None, "Should have gradients for pair input"
    assert torch.all(torch.isfinite(m_grad.grad)), "MSA gradients should be finite"
    assert torch.all(torch.isfinite(z_grad.grad)), "Pair gradients should be finite"


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=3, max_value=10),
    c=st.integers(min_value=8, max_value=32),
    n=st.integers(min_value=2, max_value=4),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test_transition(batch_size, seq_len, c, n, dtype):
    """Test Transition module."""
    device = torch.device("cpu")

    module = Transition(c=c, n=n).to(device).to(dtype)

    x = torch.randn(batch_size, seq_len, c, dtype=dtype, device=device)

    x_out = module(x)

    assert x_out.shape == x.shape, f"Expected shape {x.shape}, got {x_out.shape}"
    assert torch.all(torch.isfinite(x_out)), "Output should be finite"
    assert x_out.dtype == dtype, f"Expected dtype {dtype}, got {x_out.dtype}"

    # Test gradient computation
    x_grad = x.clone().requires_grad_(True)
    x_out_grad = module(x_grad)
    loss = x_out_grad.sum()
    loss.backward()

    assert x_grad.grad is not None, "Should have gradients for input"
    assert torch.all(torch.isfinite(x_grad.grad)), "Input gradients should be finite"

    # Test batch independence
    if batch_size > 1:
        individual_results = []
        for i in range(batch_size):
            single_input = x[i : i + 1]
            single_result = module(single_input)
            individual_results.append(single_result)

        individual_tensor = torch.cat(individual_results, dim=0)
        assert torch.allclose(x_out, individual_tensor, atol=1e-6), (
            "Batch processing should match individual processing"
        )

    # Test with different input shapes (2D input)
    x_2d = torch.randn(seq_len, c, dtype=dtype, device=device)
    x_2d_out = module(x_2d)
    assert x_2d_out.shape == x_2d.shape, "Should work with 2D input"
    assert torch.all(torch.isfinite(x_2d_out)), "2D output should be finite"
