import beignet.polynomial
import beignet.polynomial.__vander_nd
import numpy


def test__vander_nd():
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.__vander_nd._vander_nd,
        (),
        (1, 2, 3),
        [90],
    )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.__vander_nd._vander_nd,
        (),
        (),
        [90.65],
    )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.__vander_nd._vander_nd,
        (),
        (),
        [],
    )
