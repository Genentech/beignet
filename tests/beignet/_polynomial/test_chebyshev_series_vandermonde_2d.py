import beignet.polynomial
import beignet.polynomial._chebyshev_series_vandermonde_2d
import beignet.polynomial._evaluate_chebyshev_series_2d
import numpy
import torch.testing


def test_chebyshev_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))

    torch.testing.assert_close(
        torch.dot(
            beignet.polynomial.chebyshev_series_vandermonde_2d(
                x1,
                x2,
                [1, 2],
            ),
            torch.flatten(c),
        ),
        beignet.polynomial.evaluate_chebyshev_series_2d(
            x1,
            x2,
            c,
        ),
    )

    van = beignet.polynomial.chebyshev_series_vandermonde_2d(
        [x1],
        [x2],
        [1, 2],
    )

    assert van.shape == (1, 5, 6)
