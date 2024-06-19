import beignet.polynomial
import beignet.polynomial._legval
import beignet.polynomial._polyval
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_coefficients


def test_legval():
    torch.testing.assert_close(
        beignet.polynomial._legval.legval([], [1]).size,
        0,
    )

    x = numpy.linspace(-1, 1)

    y = [
        beignet.polynomial._polyval.polyval(x, c)
        for c in legendre_polynomial_coefficients
    ]

    for i in range(10):
        torch.testing.assert_close(
            beignet.polynomial._legval.legval(x, [0] * i + [1]),
            y[i],
        )

    for i in range(3):
        dims = [2] * i

        x = numpy.zeros(dims)

        torch.testing.assert_close(
            beignet.polynomial._legval.legval(x, [1]).shape,
            dims,
        )

        torch.testing.assert_close(
            beignet.polynomial._legval.legval(x, [1, 0]).shape,
            dims,
        )

        torch.testing.assert_close(
            beignet.polynomial._legval.legval(x, [1, 0, 0]).shape,
            dims,
        )
