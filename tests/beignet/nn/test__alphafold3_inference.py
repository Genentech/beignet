import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import AlphaFold3


@given(
    batch_size=st.integers(min_value=1, max_value=1),
    n_tokens=st.integers(min_value=4, max_value=8),
    dtype=st.sampled_from([torch.float32]),
)
@settings(deadline=None, max_examples=1)
def test__alphafold3_inference(batch_size, n_tokens, dtype):
    """Test AlphaFold3 with basic functionality."""

    module = AlphaFold3(
        n_cycle=1,  # Small for testing
        c_s=32,
        c_z=16,
        c_m=8,
        n_blocks_pairformer=1,  # Very small for testing
        n_head=4,
    ).to(dtype=dtype)

    # Create minimal required features matching expected interfaces
    f_star = {
        "asym_id": torch.randint(0, 3, (batch_size, n_tokens)),
        "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
        "entity_id": torch.randint(0, 2, (batch_size, n_tokens)),
        "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
        "sym_id": torch.randint(0, 5, (batch_size, n_tokens)),
        "token_bonds": torch.randn(batch_size, n_tokens, n_tokens, 32, dtype=dtype),
        "atom_coordinates": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
        "atom_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
        # Add required features for atom attention encoder
        "ref_pos": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
        "ref_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
        "ref_element": torch.randint(0, 118, (batch_size, n_tokens)),
        "ref_atom_name_chars": torch.randint(0, 26, (batch_size, n_tokens, 4)),
        "ref_charge": torch.randn(batch_size, n_tokens, dtype=dtype),
        "restype": torch.randint(0, 21, (batch_size, n_tokens)),
        "profile": torch.randn(batch_size, n_tokens, 20, dtype=dtype),
        "deletion_mean": torch.randn(batch_size, n_tokens, dtype=dtype),
        "ref_space_uid": torch.randint(0, 1000, (batch_size, n_tokens)),
    }

    # Test forward pass
    outputs = module(f_star)

    # Check output structure
    assert isinstance(outputs, dict)
    required_keys = ["x_pred", "p_plddt", "p_pae", "p_pde", "p_resolved", "p_distogram"]
    for key in required_keys:
        assert key in outputs, f"Missing output key: {key}"
        assert torch.all(torch.isfinite(outputs[key]))

    # Basic shape and type checks
    assert outputs["x_pred"].shape == (batch_size, n_tokens, 3)
    assert outputs["x_pred"].dtype == dtype
