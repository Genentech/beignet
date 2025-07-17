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
    parameters, _ = data

    rotation_matrices = beignet.random_rotation_matrix(**parameters)

    assert rotation_matrices.shape == (parameters["size"], 3, 3)

    determinants = torch.det(rotation_matrices)
    torch.testing.assert_close(
        determinants, torch.ones_like(determinants), atol=1e-5, rtol=1e-5
    )
