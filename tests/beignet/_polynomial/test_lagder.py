import beignet.polynomial
import numpy
import torch


def test_lagder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial.lagder(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial.lagtrim(res, tolerance=1e-6),
            beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.lagder(beignet.polynomial.lagint(tgt, m=j), m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.lagder(
                beignet.polynomial.lagint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.lagder(c) for c in c2d.T]).T
    res = beignet.polynomial.lagder(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.lagder(c) for c in c2d])
    res = beignet.polynomial.lagder(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)
