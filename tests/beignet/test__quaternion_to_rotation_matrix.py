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

    return (
        {
            "input": torch.from_numpy(
                rotation.as_quat(
                    canonical=False,
                ),
            ),
        },
        torch.from_numpy(
            rotation.as_matrix(),
        ),
    )


@hypothesis.given(_strategy())
def test_quaternion_to_rotation_matrix(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.quaternion_to_rotation_matrix(
            **parameters,
        ),
        expected,
    )
