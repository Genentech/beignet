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

    return (
        {
            "input": torch.from_numpy(input.as_matrix()),
            "other": torch.from_numpy(other.as_matrix()),
        },
        torch.from_numpy(operator.mul(input, other).as_matrix()),
    )


@hypothesis.given(_strategy())
def test_compose_rotation_matrix(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.compose_rotation_matrix(**parameters),
        expected,
    )
