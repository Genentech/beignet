import hypothesis.extra.numpy
import hypothesis.strategies
import torch
from scipy.spatial.transform import Rotation

import beignet


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

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(
                rotation.as_rotvec(
                    degrees,
                ),
            ),
            "degrees": degrees,
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
def test_rotation_vector_to_quaternion(data):
    parameters, expected = data

    torch.testing.assert_close(
        torch.abs(
            beignet.rotation_vector_to_quaternion(
                **parameters,
            ),
        ),
        expected,
    )
