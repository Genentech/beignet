import beignet.polynomial
import numpy


def test__trim_coefficients():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._trim_coefficients, coef, -1
    )

    numpy.testing.assert_equal(
        beignet.polynomial._trim_coefficients(coef),
        coef[:-1],
    )

    numpy.testing.assert_equal(
        beignet.polynomial._trim_coefficients(coef, 1),
        coef[:-3],
    )

    numpy.testing.assert_equal(
        beignet.polynomial._trim_coefficients(coef, 2),
        [0],
    )
