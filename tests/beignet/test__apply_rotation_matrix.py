import beignet
import hypothesis.extra.numpy
import hypothesis.strategies
import numpy
import torch.testing
from scipy.spatial.transform import Rotation


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

    rotation = Rotation.random(size)

    inverse = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(
                input,
            ),
            "rotation": torch.from_numpy(
                rotation.as_matrix(),
            ),
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
def test_apply_rotation_matrix(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.apply_rotation_matrix(
            **parameters,
        ),
        expected,
        equal_nan=True,
        atol=1e-06,
        rtol=1e-06,
    )
