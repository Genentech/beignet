import beignet
import hypothesis.extra.numpy
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

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(
                rotation.as_matrix(),
            ),
            "canonical": canonical,
        },
        torch.abs(
            torch.from_numpy(
                rotation.as_quat(
                    canonical,
                ),
            ),
        ),
    )


@hypothesis.given(_strategy())
def test_rotation_matrix_to_quaternion(data):
    parameters, expected = data

    torch.testing.assert_close(
        torch.abs(
            beignet.rotation_matrix_to_quaternion(
                **parameters,
            ),
        ),
        expected,
    )
