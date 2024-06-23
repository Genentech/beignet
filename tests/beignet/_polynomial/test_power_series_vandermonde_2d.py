import beignet.polynomial
import numpy
import torch.testing


def test_power_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random([3, 5]) * 2 - 1

    c = numpy.random.random([2, 3])

    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x3 = torch.from_numpy(x3)

    c = torch.from_numpy(c)

    torch.testing.assert_close(
        torch.dot(
            beignet.polynomial.power_series_vandermonde_2d(
                x1,
                x2,
                [1, 2],
            ),
            torch.flatten(c),
        ),
        beignet.polynomial.evaluate_power_series_2d(
            x1,
            x2,
            c,
        ),
    )

    # assert beignet.polynomial.power_series_vandermonde_2d(
    #     [x1],
    #     [x2],
    #     [1, 2],
    # ).shape == (1, 5, 6)
