import beignet.polynomial
import numpy


def test__pow():
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._pow,
        (),
        [1, 2, 3],
        5,
        4,
    )
