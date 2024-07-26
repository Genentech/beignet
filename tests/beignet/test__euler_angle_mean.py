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
            "axes": axes,
            "degrees": degrees,
        },
        torch.unsqueeze(
            torch.from_numpy(
                input.mean().as_euler(axes, degrees),
            ),
            dim=0,
        ),
    )


@hypothesis.given(_strategy())
def test_euler_angle_mean(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.euler_angle_mean(**parameters),
        expected,
    )
