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

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(input.as_quat(canonical)),
            "other": torch.from_numpy(other.as_quat(canonical)),
            "canonical": canonical,
        },
        torch.abs(torch.from_numpy(operator.mul(input, other).as_quat(canonical))),
    )


@hypothesis.given(_strategy())
def test_compose_quaternion(data):
    parameters, expected = data

    torch.testing.assert_close(
        torch.abs(beignet.compose_quaternion(**parameters)),
        expected,
    )
