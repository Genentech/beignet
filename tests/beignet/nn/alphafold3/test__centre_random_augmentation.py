import torch

from beignet.nn.alphafold3._sample_diffusion import _CentreRandomAugmentation


def test__centre_random_augmentation():
    """Test CentreRandomAugmentation with fixed input configurations."""

    # Use fixed parameters to avoid flakiness
    batch_size = 2
    n_atoms = 8
    s_trans = 1.0
    dtype = torch.float32
    device = torch.device("cpu")

    module = _CentreRandomAugmentation(s_trans=s_trans)

    # Create test input positions
    x_t = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)

    # Test forward pass
    output = module(x_t)

    # Check output shape
    assert output.shape == (batch_size, n_atoms, 3)
    assert output.dtype == dtype

    # Check that output is finite
    assert torch.all(torch.isfinite(output))

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

    # Test that rotation matrix is orthogonal
    # (We can't directly test this from the output, but we can test the generator)
    R = module.uniform_random_rotation(2, x_t.device, dtype)

    # Check that R is orthogonal: R @ R.T = I
    i = torch.bmm(R, R.transpose(-2, -1))
    expected_I = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(i)
    assert torch.allclose(i, expected_I, atol=1e-5)

    # Check that det(R) = 1 (proper rotation, not reflection)
    det_R = torch.det(R)
    assert torch.allclose(det_R, torch.ones_like(det_R), atol=1e-5)

    # Test edge cases
    # Single atom
    x_single = torch.randn(batch_size, 1, 3, dtype=dtype, device=device)
    output_single = module(x_single)
    assert output_single.shape == (batch_size, 1, 3)

    # Two atoms - should preserve distance
    x_two = torch.randn(batch_size, 2, 3, dtype=dtype, device=device)
    output_two = module(x_two)
    assert output_two.shape == (batch_size, 2, 3)

    # Check distance preservation for two atoms
    orig_dist_two = torch.norm(x_two[:, 0] - x_two[:, 1], dim=-1)
    new_dist_two = torch.norm(output_two[:, 0] - output_two[:, 1], dim=-1)
    assert torch.allclose(orig_dist_two, new_dist_two, atol=1e-5)

    # Test different s_trans values produce different translation scales
    module_large = _CentreRandomAugmentation(s_trans=10.0)
    output_large = module_large(x_t)

    # Outputs should be finite
    assert torch.all(torch.isfinite(output_large))

    # Test with zero s_trans (no translation)
    module_zero = _CentreRandomAugmentation(s_trans=0.0)
    output_zero = module_zero(x_t)
    assert torch.all(torch.isfinite(output_zero))

    # With zero translation, the mean should be approximately zero after centering
    output_zero_mean = output_zero.mean(dim=1)  # (batch_size, 3)
    mean_norms = torch.norm(output_zero_mean, dim=-1)
    # With s_trans=0, translation should be zero, so mean norm should be small
    assert torch.all(mean_norms < 1e-6), (
        f"Mean norms should be small with s_trans=0, got {mean_norms}"
    )
