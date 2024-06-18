import beignet.polynomial
import numpy


def test__div():
    numpy.testing.assert_raises(
        ZeroDivisionError,
        beignet.polynomial._div,
        beignet.polynomial._div,
        (1, 2, 3),
        [0],
    )
