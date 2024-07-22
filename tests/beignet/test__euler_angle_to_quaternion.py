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

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(
                rotation.as_euler(
                    axes,
                    degrees,
                ),
            ),
            "axes": axes,
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
def test_euler_angle_to_quaternion(data):
    parameters, expected = data

    torch.testing.assert_close(
        torch.abs(
            beignet.euler_angle_to_quaternion(
                **parameters,
            ),
        ),
        expected,
    )
