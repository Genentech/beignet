import beignet.polynomial
import beignet.polynomial._evaluate_1d_physicists_hermite_series
import beignet.polynomial._evaluate_1d_power_series
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import hermite_polynomial_coefficients


def test_evaluate_1d_physicists_hermite_series():
    torch.testing.assert_close(
        beignet.polynomial._hermval.evaluate_1d_physicists_hermite_series([], [1]).size,
        0,
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial._polyval.evaluate_1d_power_series(x, c)
        for c in hermite_polynomial_coefficients
    ]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial._hermval.evaluate_1d_physicists_hermite_series(
            x, [0] * i + [1]
        )
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial._hermval.evaluate_1d_physicists_hermite_series(
                x, [1]
            ).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial._hermval.evaluate_1d_physicists_hermite_series(
                x, [1, 0]
            ).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial._hermval.evaluate_1d_physicists_hermite_series(
                x, [1, 0, 0]
            ).shape,
            dims,
        )
