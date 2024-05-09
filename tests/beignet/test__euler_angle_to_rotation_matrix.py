import beignet
import hypothesis.strategies
import torch.testing
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
        },
        torch.from_numpy(
            rotation.as_matrix(),
        ),
    )


@hypothesis.given(_strategy())
def test_euler_angle_to_rotation_matrix(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.euler_angle_to_rotation_matrix(
            **parameters,
        ),
        expected,
    )
