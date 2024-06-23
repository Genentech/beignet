import numpy
import torch
from beignet.polynomial import (
    evaluate_probabilists_hermite_series_2d,
    probabilists_hermite_series_vandermonde_2d,
)


def test_probabilists_hermite_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    torch.testing.assert_close(
        torch.dot(
            probabilists_hermite_series_vandermonde_2d(x1, x2, [1, 2]),
            torch.flatten(c),
        ),
        evaluate_probabilists_hermite_series_2d(x1, x2, c),
    )

    output = probabilists_hermite_series_vandermonde_2d([x1], [x2], [1, 2])

    assert output.shape == (1, 5, 6)
