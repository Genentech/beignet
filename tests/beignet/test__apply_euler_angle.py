import hypothesis.extra.numpy
import hypothesis.strategies
import numpy
import torch.testing
from scipy.spatial.transform import Rotation

import beignet


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    input = function(
        hypothesis.extra.numpy.arrays(
            numpy.float64,
            (size, 3),
            elements={
                "allow_infinity": False,
                "min_value": numpy.finfo(numpy.float32).min,
                "max_value": numpy.finfo(numpy.float32).max,
            },
        ),
    )

    rotation = Rotation.random(
        size,
    )

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

    degrees = function(
        hypothesis.strategies.booleans(),
    )

    inverse = function(
        hypothesis.strategies.booleans(),
    )

    return (
        {
            "input": torch.from_numpy(
                input,
            ),
            "rotation": torch.from_numpy(
                rotation.as_euler(
                    axes,
                    degrees,
                ),
            ),
            "axes": axes,
            "degrees": degrees,
            "inverse": inverse,
        },
        torch.from_numpy(
            rotation.apply(
                input,
                inverse,
            ),
        ),
    )


@hypothesis.given(_strategy())
def test_apply_euler_angle(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.apply_euler_angle(
            **parameters,
        ),
        expected,
    )
