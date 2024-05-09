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
                rotation.as_quat(
                    canonical=False,
                ),
            ),
            "degrees": degrees,
        },
        torch.from_numpy(
            rotation.as_rotvec(
                degrees,
            ),
        ),
    )


@hypothesis.given(_strategy())
def test_quaternion_to_rotation_vector(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.quaternion_to_rotation_vector(
            **parameters,
        ),
        expected,
    )
