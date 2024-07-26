import beignet
import hypothesis.strategies


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
        None,
    )


@hypothesis.given(_strategy())
def test_random_euler_angle(data):
    parameters, _ = data

    assert beignet.random_euler_angle(
        **parameters,
    ).shape == (parameters["size"], 3)
