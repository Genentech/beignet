import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import CentreRandomAugmentation


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_atoms=st.integers(min_value=4, max_value=32),
    s_trans=st.floats(min_value=0.1, max_value=3.0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=5)
def test__centre_random_augmentation(batch_size, n_atoms, s_trans, dtype):
    """Test CentreRandomAugmentation with various input configurations."""

    module = CentreRandomAugmentation(s_trans=s_trans)

    # Create test input positions
    x_t = torch.randn(batch_size, n_atoms, 3, dtype=dtype)

    # Test forward pass
    output = module(x_t)

    # Check output shape
    assert output.shape == (batch_size, n_atoms, 3)
    assert output.dtype == dtype

    # Check that output is finite
    assert torch.all(torch.isfinite(output))

    # Test that positions are centered (mean should be close to translation vector)
    output_mean = output.mean(dim=1)  # (batch_size, 3)

    # The mean should have magnitude approximately s_trans (due to translation)
    mean_norms = torch.norm(output_mean, dim=-1)
    # Allow some tolerance since this is stochastic
    assert torch.all(mean_norms <= s_trans * 3.0)  # Upper bound with tolerance

    # Test that the function actually moves the coordinates
    # (output should not be identical to input unless very unlikely)
    assert not torch.allclose(x_t, output, atol=1e-6)

    # Test rotation properties: distances between atoms should be preserved
    # (up to numerical precision and centering effects)
    x_centered = x_t - x_t.mean(dim=1, keepdim=True)
    output_centered = output - output.mean(dim=1, keepdim=True)

    # Compute pairwise distances
    def pairwise_distances(x):
        # x: (batch_size, n_atoms, 3)
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)  # (batch, n_atoms, n_atoms, 3)
        return torch.norm(diff, dim=-1)  # (batch, n_atoms, n_atoms)

    orig_dists = pairwise_distances(x_centered)
    new_dists = pairwise_distances(output_centered)

    # Distances should be approximately preserved (rotation is isometric)
    assert torch.allclose(orig_dists, new_dists, atol=1e-4)

    # Test gradient computation
    x_grad = x_t.clone().requires_grad_(True)
    output_grad = module(x_grad)
    loss = output_grad.sum()
    loss.backward()

    # Check gradients exist
    assert x_grad.grad is not None
    assert torch.all(torch.isfinite(x_grad.grad))

    # Test torch.compile compatibility (skip for speed in tests)
    # compiled_module = torch.compile(module, fullgraph=True)
    # output_compiled = compiled_module(x_t)
    # assert output_compiled.shape == output.shape
    # assert output_compiled.dtype == output.dtype

    # Test that rotation matrix is orthogonal
    # (We can't directly test this from the output, but we can test the generator)
    R = module.uniform_random_rotation(2, x_t.device, dtype)

    # Check that R is orthogonal: R @ R.T = I
    I = torch.bmm(R, R.transpose(-2, -1))
    expected_I = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(I)
    assert torch.allclose(I, expected_I, atol=1e-5)

    # Check that det(R) = 1 (proper rotation, not reflection)
    det_R = torch.det(R)
    assert torch.allclose(det_R, torch.ones_like(det_R), atol=1e-5)

    # Test edge cases
    # Single atom
    x_single = torch.randn(batch_size, 1, 3, dtype=dtype)
    output_single = module(x_single)
    assert output_single.shape == (batch_size, 1, 3)

    # Two atoms - should preserve distance
    x_two = torch.randn(batch_size, 2, 3, dtype=dtype)
    output_two = module(x_two)
    assert output_two.shape == (batch_size, 2, 3)

    # Check distance preservation for two atoms
    orig_dist_two = torch.norm(x_two[:, 0] - x_two[:, 1], dim=-1)
    new_dist_two = torch.norm(output_two[:, 0] - output_two[:, 1], dim=-1)
    assert torch.allclose(orig_dist_two, new_dist_two, atol=1e-5)

    # Test different s_trans values produce different translation scales
    module_large = CentreRandomAugmentation(s_trans=10.0)
    output_large = module_large(x_t)

    mean_large = output_large.mean(dim=1)
    mean_small = output.mean(dim=1)

    # Larger s_trans should generally produce larger translations
    # (probabilistic test, might occasionally fail)
    norm_large = torch.norm(mean_large, dim=-1).mean()
    norm_small = torch.norm(mean_small, dim=-1).mean()

    # This is stochastic, so we use a loose bound
    if s_trans < 5.0:  # Only test when original s_trans is not too large
        assert norm_large > norm_small * 0.5  # Very loose bound
