import hypothesis.extra.numpy
import hypothesis.strategies
import numpy
import torch.testing

import beignet


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=128,
            max_value=512,
        ),
    )

    input = torch.rand([size])
    other = torch.rand([size])

    return (
        {
            "input": input,
            "other": other,
        },
        torch.reshape(
            torch.from_numpy(numpy.convolve(input, other)),
            [1, -1],
        ),
    )


@hypothesis.given(_strategy())
def test_convolve(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.convolve(**parameters),
        expected,
    )
