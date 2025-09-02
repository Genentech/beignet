import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import AlphaFold3Distogram


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    n_tokens=st.integers(min_value=8, max_value=16),
    c_z=st.integers(min_value=8, max_value=32),
    n_bins=st.integers(min_value=16, max_value=64),
    dtype=st.sampled_from([torch.float32]),
)
@settings(deadline=None, max_examples=3)
def test__distogram_head(batch_size, n_tokens, c_z, n_bins, dtype):
    """Test AlphaFold3Distogram with various input configurations."""

    module = AlphaFold3Distogram(
        c_z=c_z,
        n_bins=n_bins,
        min_dist=2.0,
        max_dist=20.0,
    ).to(dtype=dtype)

    # Create test inputs
    z_ij = torch.randn(batch_size, n_tokens, n_tokens, c_z, dtype=dtype)

    # Test forward pass
    p_distogram = module(z_ij)

    # Check output shape
    assert p_distogram.shape == (batch_size, n_tokens, n_tokens, n_bins)
    assert p_distogram.dtype == dtype

    # Check that output is finite
    assert torch.all(torch.isfinite(p_distogram))

    # Check that probabilities are non-negative and sum to 1
    assert torch.all(p_distogram >= 0)
    assert torch.all(p_distogram <= 1)

    # Check that probabilities sum to 1 over the bin dimension
    prob_sums = torch.sum(p_distogram, dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    # Test gradient computation
    z_grad = z_ij.clone().requires_grad_(True)
    output_grad = module(z_grad)
    loss = output_grad.sum()
    loss.backward()

    # Check gradients exist
    assert z_grad.grad is not None
    assert torch.all(torch.isfinite(z_grad.grad))

    # Test that different inputs produce different outputs
    z_ij_2 = torch.randn(batch_size, n_tokens, n_tokens, c_z, dtype=dtype)
    p_distogram_2 = module(z_ij_2)

    # Different inputs should generally produce different outputs
    assert not torch.allclose(p_distogram, p_distogram_2, atol=1e-3)

    # Test distance bins are properly registered
    assert hasattr(module, "distance_bins")
    assert module.distance_bins.shape == (n_bins,)
    assert module.distance_bins[0] >= 2.0  # min_dist
    assert module.distance_bins[-1] <= 20.0  # max_dist

    # Test different number of bins
    module_small = AlphaFold3Distogram(c_z=c_z, n_bins=8).to(dtype=dtype)
    p_small = module_small(z_ij)
    assert p_small.shape == (batch_size, n_tokens, n_tokens, 8)

    # Test edge case with single token
    if n_tokens >= 1:
        z_single = torch.randn(batch_size, 1, 1, c_z, dtype=dtype)
        p_single = module(z_single)
        assert p_single.shape == (batch_size, 1, 1, n_bins)
        assert torch.all(torch.isfinite(p_single))

        # Check probabilities still sum to 1
        prob_sums_single = torch.sum(p_single, dim=-1)
        assert torch.allclose(
            prob_sums_single, torch.ones_like(prob_sums_single), atol=1e-5
        )
