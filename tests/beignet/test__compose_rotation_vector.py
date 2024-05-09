import operator

import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=16,
            max_value=32,
        ),
    )

    input = Rotation.random(size)
    other = Rotation.random(size)

    degrees = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(input.as_rotvec(degrees)),
            "other": torch.from_numpy(other.as_rotvec(degrees)),
            "degrees": degrees,
        },
        torch.from_numpy(operator.mul(input, other).as_rotvec(degrees)),
    )


@hypothesis.given(_strategy())
def test_compose_rotation_vector(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.compose_rotation_vector(**parameters),
        expected,
    )
