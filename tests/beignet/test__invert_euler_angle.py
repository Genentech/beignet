import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
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

    rotations = Rotation.random(
        function(
            hypothesis.strategies.integers(
                min_value=1,
                max_value=8,
            ),
        ),
    )

    return (
        {
            "input": torch.from_numpy(rotations.as_euler(axes, degrees)),
            "axes": axes,
            "degrees": degrees,
        },
        torch.from_numpy(rotations.inv().as_euler(axes, degrees)),
    )


@hypothesis.given(_strategy())
def test_invert_euler_angle(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.invert_euler_angle(**parameters),
        expected,
    )
