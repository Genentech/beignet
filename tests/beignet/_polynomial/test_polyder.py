import beignet.polynomial
import numpy


def test_polyder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.polyder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polyder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        numpy.testing.assert_equal(
            beignet.polynomial.polytrim(
                beignet.polynomial.polyder(tgt, order=0), tolerance=1e-6
            ),
            beignet.polynomial.polytrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.polyder(
                beignet.polynomial.polyint(tgt, m=j), order=j
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.polyder(
                beignet.polynomial.polyint(tgt, m=j, scl=2), order=j, scale=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.polyder(c) for c in c2d.T]).T
    numpy.testing.assert_almost_equal(beignet.polynomial.polyder(c2d, axis=0), tgt)

    tgt = numpy.vstack([beignet.polynomial.polyder(c) for c in c2d])
    numpy.testing.assert_almost_equal(beignet.polynomial.polyder(c2d, axis=1), tgt)
