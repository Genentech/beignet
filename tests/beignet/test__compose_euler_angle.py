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

    axes = function(
        hypothesis.strategies.sampled_from(
            [
                "xyz",
                "xzy",
                "yxz",
                "yzx",
                "zxy",
                "zyx",
                "XYZ",
                "XZY",
                "YXZ",
                "YZX",
                "ZXY",
                "ZYX",
            ]
        ),
    )

    degrees = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(input.as_euler(axes, degrees)),
            "other": torch.from_numpy(other.as_euler(axes, degrees)),
            "axes": axes,
            "degrees": degrees,
        },
        torch.from_numpy(operator.mul(input, other).as_euler(axes, degrees)),
    )


@hypothesis.given(_strategy())
def test_compose_euler_angle(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.compose_euler_angle(**parameters),
        expected,
    )
