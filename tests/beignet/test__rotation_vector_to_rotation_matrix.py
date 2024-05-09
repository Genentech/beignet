import beignet
import hypothesis.strategies
import torch
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    rotation = Rotation.random(
        function(
            hypothesis.strategies.integers(
                min_value=16,
                max_value=32,
            ),
        ),
    )

    degrees = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(
                rotation.as_rotvec(degrees),
            ),
            "degrees": degrees,
        },
        torch.from_numpy(
            rotation.as_matrix(),
        ),
    )


@hypothesis.given(_strategy())
def test_rotation_vector_to_rotation_matrix(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_vector_to_rotation_matrix(
            **parameters,
        ),
        expected,
    )
