import beignet.polynomial
import numpy


def test__vander_nd():
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._vander_nd,
        (),
        (1, 2, 3),
        [90],
    )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._vander_nd,
        (),
        (),
        [90.65],
    )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._vander_nd,
        (),
        (),
        [],
    )
