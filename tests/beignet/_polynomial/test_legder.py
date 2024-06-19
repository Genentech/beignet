import beignet.polynomial
import numpy
import torch


def test_legder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial.legder(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial.legtrim(res, tolerance=1e-6),
            beignet.polynomial.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.legder(beignet.polynomial.legint(tgt, m=j), m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.legder(
                beignet.polynomial.legint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.legder(c) for c in c2d.T]).T
    res = beignet.polynomial.legder(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.legder(c) for c in c2d])
    res = beignet.polynomial.legder(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    c = (1, 2, 3, 4)
    torch.testing.assert_close(beignet.polynomial.legder(c, 4), [0])
