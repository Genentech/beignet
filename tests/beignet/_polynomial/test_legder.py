import beignet.polynomial
import beignet.polynomial._legder
import beignet.polynomial._legint
import beignet.polynomial._legtrim
import numpy
import torch


def test_legder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial._legder.legder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial._legder.legder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial._legder.legder(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
            beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial._legder.legder(
                beignet.polynomial._legint.legint(tgt, m=j), m=j
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial._legder.legder(
                beignet.polynomial._legint.legint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial._legder.legder(c) for c in c2d.T]).T
    res = beignet.polynomial._legder.legder(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial._legder.legder(c) for c in c2d])
    res = beignet.polynomial._legder.legder(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    c = (1, 2, 3, 4)
    torch.testing.assert_close(beignet.polynomial._legder.legder(c, 4), [0])
