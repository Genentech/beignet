import beignet
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

    return (
        {
            "input": torch.from_numpy(
                rotation.as_matrix(),
            ),
            "axes": axes,
            "degrees": degrees,
        },
        torch.from_numpy(
            rotation.as_euler(
                axes,
                degrees,
            ),
        ),
    )


@hypothesis.given(_strategy())
def test_rotation_matrix_to_euler_angle(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_matrix_to_euler_angle(
            **parameters,
        ),
        expected,
    )
