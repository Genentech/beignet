import hypothesis.strategies
import torch

import beignet


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    return (
        {
            "size": size,
        },
        None,
    )


@hypothesis.given(_strategy())
def test_random_rotation_matrix(data):
    """Test that random_rotation_matrix generates valid rotation matrices with correct properties."""
    parameters, _ = data

    rotation_matrices = beignet.random_rotation_matrix(**parameters)

    # Test shape
    assert rotation_matrices.shape == (parameters["size"], 3, 3)

    # Test orthogonality: R @ R.T = I
    identity = torch.eye(
        3, dtype=rotation_matrices.dtype, device=rotation_matrices.device
    )
    identity_batch = identity.expand(parameters["size"], 3, 3)
    orthogonal_product = torch.bmm(
        rotation_matrices, rotation_matrices.transpose(-1, -2)
    )
    torch.testing.assert_close(orthogonal_product, identity_batch, atol=1e-5, rtol=1e-5)

    # Test determinant = 1
    determinants = torch.det(rotation_matrices)
    torch.testing.assert_close(
        determinants, torch.ones_like(determinants), atol=1e-5, rtol=1e-5
    )

    # Test distance preservation
    test_vectors = torch.randn(parameters["size"], 3)
    rotated_vectors = torch.bmm(rotation_matrices, test_vectors.unsqueeze(-1)).squeeze(
        -1
    )
    original_norms = torch.norm(test_vectors, dim=-1)
    rotated_norms = torch.norm(rotated_vectors, dim=-1)
    torch.testing.assert_close(rotated_norms, original_norms, atol=1e-5, rtol=1e-5)
