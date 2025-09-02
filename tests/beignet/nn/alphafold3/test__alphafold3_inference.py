import torch

from beignet.nn.alphafold3 import AlphaFold3


def test__alphafold3_inference():
    """Test AlphaFold3 basic instantiation and forward compatibility."""
    dtype = torch.float32

    # Create minimal module configuration for testing
    module = AlphaFold3(
        n_cycle=1,
        c_s=16,
        c_z=8,
        c_m=8,  # Must be divisible by n_head
        n_blocks_pairformer=1,
        n_head=8,  # Large enough so n_head//4 >= 1
    ).to(dtype=dtype)

    # Test basic module properties
    assert hasattr(module, "input_feature_embedder")
    assert hasattr(module, "pairformer_stack")
    assert hasattr(module, "msa_module")
    assert hasattr(module, "sample_diffusion")
    assert hasattr(module, "confidence_head")
    assert hasattr(module, "distogram_head")

    # Test module parameters
    param_count = sum(p.numel() for p in module.parameters())
    assert param_count > 0, "Module should have trainable parameters"

    # Test that module components are properly initialized
    for name, param in module.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} should be finite"
        assert param.dtype == dtype, f"Parameter {name} should have correct dtype"
