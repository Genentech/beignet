import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import _Confidence


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    n_atoms=st.integers(min_value=4, max_value=16),
    n_tokens=st.integers(min_value=2, max_value=8),
    c_s=st.integers(min_value=16, max_value=32).filter(
        lambda x: x % 16 == 0
    ),  # Divisible by 16 heads
    c_z=st.integers(min_value=8, max_value=16).filter(
        lambda x: x % 4 == 0
    ),  # Divisible by 4 heads
    dtype=st.sampled_from([torch.float32]),  # Only float32 for simplicity
)
@settings(deadline=None, max_examples=3)
def test__confidence_head(batch_size, n_atoms, n_tokens, c_s, c_z, dtype):
    """Test AlphaFold3Confidence with various input configurations."""

    # Ensure n_atoms is divisible by or larger than n_tokens for testing
    n_atoms = max(n_atoms, n_tokens)

    module = _Confidence(
        n_block=2,  # Use smaller number for testing
        c_s=c_s,
        c_z=c_z,
    ).to(dtype=dtype)

    # Create test inputs
    s_inputs = torch.randn(batch_size, n_atoms, 100, dtype=dtype)  # 100 features
    s_i = torch.randn(batch_size, n_tokens, c_s, dtype=dtype)
    z_ij = torch.randn(batch_size, n_tokens, n_tokens, c_z, dtype=dtype)
    x_pred = torch.randn(batch_size, n_atoms, 3, dtype=dtype)

    # Test forward pass
    p_plddt, p_pae, p_pde, p_resolved = module(s_inputs, s_i, z_ij, x_pred)

    # Check output shapes
    assert p_plddt.shape == (batch_size, n_atoms, 50)  # 50 pLDDT bins
    assert p_pae.shape == (batch_size, n_tokens, n_tokens, 64)  # 64 PAE bins
    assert p_pde.shape == (batch_size, n_tokens, n_tokens, 64)  # 64 PDE bins
    assert p_resolved.shape == (batch_size, n_atoms, 2)  # 2 resolved classes

    # Check output dtypes
    assert p_plddt.dtype == dtype
    assert p_pae.dtype == dtype
    assert p_pde.dtype == dtype
    assert p_resolved.dtype == dtype

    # Check that outputs are finite
    assert torch.all(torch.isfinite(p_plddt))
    assert torch.all(torch.isfinite(p_pae))
    assert torch.all(torch.isfinite(p_pde))
    assert torch.all(torch.isfinite(p_resolved))

    # Check that outputs are proper probability distributions (sum to 1)
    assert torch.allclose(
        p_plddt.sum(dim=-1), torch.ones(batch_size, n_atoms, dtype=dtype), atol=1e-5
    )
    assert torch.allclose(
        p_pae.sum(dim=-1),
        torch.ones(batch_size, n_tokens, n_tokens, dtype=dtype),
        atol=1e-5,
    )
    assert torch.allclose(
        p_pde.sum(dim=-1),
        torch.ones(batch_size, n_tokens, n_tokens, dtype=dtype),
        atol=1e-5,
    )
    assert torch.allclose(
        p_resolved.sum(dim=-1), torch.ones(batch_size, n_atoms, dtype=dtype), atol=1e-5
    )

    # Check that probabilities are non-negative
    assert torch.all(p_plddt >= 0)
    assert torch.all(p_pae >= 0)
    assert torch.all(p_pde >= 0)
    assert torch.all(p_resolved >= 0)

    # Test gradient computation
    s_inputs_grad = s_inputs.clone().requires_grad_(True)
    s_i_grad = s_i.clone().requires_grad_(True)
    z_ij_grad = z_ij.clone().requires_grad_(True)
    x_pred_grad = x_pred.clone().requires_grad_(True)

    p_plddt_grad, p_pae_grad, p_pde_grad, p_resolved_grad = module(
        s_inputs_grad, s_i_grad, z_ij_grad, x_pred_grad
    )

    # Create dummy loss
    loss = (
        p_plddt_grad.sum() + p_pae_grad.sum() + p_pde_grad.sum() + p_resolved_grad.sum()
    )
    loss.backward()

    # Check gradients exist
    assert s_inputs_grad.grad is not None
    assert s_i_grad.grad is not None
    assert z_ij_grad.grad is not None
    assert x_pred_grad.grad is not None
    assert torch.all(torch.isfinite(s_inputs_grad.grad))

    # Test torch.compile compatibility (skip for speed in tests)
    # compiled_module = torch.compile(module, fullgraph=True)
    # p_plddt_comp, p_pae_comp, p_pde_comp, p_resolved_comp = compiled_module(
    #     s_inputs, s_i, z_ij, x_pred
    # )
    # assert torch.allclose(p_plddt, p_plddt_comp, atol=1e-5)
    # assert torch.allclose(p_pae, p_pae_comp, atol=1e-5)
    # assert torch.allclose(p_pde, p_pde_comp, atol=1e-5)
    # assert torch.allclose(p_resolved, p_resolved_comp, atol=1e-5)

    # Test that different inputs produce different outputs
    s_inputs_2 = torch.randn(batch_size, n_atoms, 100, dtype=dtype)
    p_plddt_2, p_pae_2, p_pde_2, p_resolved_2 = module(s_inputs_2, s_i, z_ij, x_pred)

    # Different inputs should generally produce different outputs
    assert not torch.allclose(p_plddt, p_plddt_2, atol=1e-3)

    # Test PDE symmetry property: PDE uses z_ij + z_ji
    # So it should have some relationship to the transpose
    # (Though it's not exactly symmetric due to the linear layers)

    # Test distance embedding behavior
    # Closer atoms should have different confidence patterns than distant ones
    x_close = torch.zeros(batch_size, n_atoms, 3, dtype=dtype)  # All atoms at origin
    x_far = torch.randn(batch_size, n_atoms, 3, dtype=dtype) * 10  # Spread out atoms

    _, p_pae_close, _, _ = module(s_inputs, s_i, z_ij, x_close)
    _, p_pae_far, _, _ = module(s_inputs, s_i, z_ij, x_far)

    # Different position configurations should produce different PAE predictions
    assert not torch.allclose(p_pae_close, p_pae_far, atol=1e-3)

    # Test edge cases
    # Single token case
    if n_tokens >= 1:
        s_i_single = torch.randn(batch_size, 1, c_s, dtype=dtype)
        z_ij_single = torch.randn(batch_size, 1, 1, c_z, dtype=dtype)

        p_plddt_single, p_pae_single, p_pde_single, p_resolved_single = module(
            s_inputs, s_i_single, z_ij_single, x_pred
        )

        assert p_pae_single.shape == (batch_size, 1, 1, 64)
        assert p_pde_single.shape == (batch_size, 1, 1, 64)

    # Test that pLDDT and resolved outputs depend on atom-level features
    # (They should vary when we change s_inputs while keeping others constant)
    s_inputs_alt = s_inputs + 0.1  # Small perturbation
    p_plddt_alt, _, _, p_resolved_alt = module(s_inputs_alt, s_i, z_ij, x_pred)

    assert not torch.allclose(p_plddt, p_plddt_alt, atol=1e-4)
    assert not torch.allclose(p_resolved, p_resolved_alt, atol=1e-4)
