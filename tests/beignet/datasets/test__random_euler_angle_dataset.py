import beignet
import hypothesis.strategies
from beignet.datasets import RandomEulerAngleDataset


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
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
            "size": size,
            "axes": axes,
            "degrees": degrees,
        },
        beignet.random_euler_angle(size, axes=axes, degrees=degrees),
    )


class TestRandomEulerAngleDataset:
    @hypothesis.given(_strategy())
    def test___init__(self, data):
        parameters, output = data

        dataset = RandomEulerAngleDataset(**parameters)

        assert dataset.data.shape == output.shape

        assert dataset.data.dtype == output.dtype

        assert dataset.data.layout == output.layout
