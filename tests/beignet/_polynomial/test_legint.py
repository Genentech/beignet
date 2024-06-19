import beignet.polynomial
import beignet.polynomial._legtrim
import numpy
import torch


def test_legint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legint([0], m=i, k=k), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        legpol = beignet.polynomial.poly2leg(pol)
        legint = beignet.polynomial.legint(legpol, m=1, k=[i])
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legtrim.legtrim(
                beignet.polynomial.leg2poly(legint), tolerance=1e-6
            ),
            beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        legpol = beignet.polynomial.poly2leg(pol)
        legint = beignet.polynomial.legint(legpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.legval(-1, legint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        legpol = beignet.polynomial.poly2leg(pol)
        legint = beignet.polynomial.legint(legpol, m=1, k=[i], scl=2)
        res = beignet.polynomial.leg2poly(legint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
            beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1)
            res = beignet.polynomial.legint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1, k=[k])
            res = beignet.polynomial.legint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.legint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1, k=[k], scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.legtrim(
                    beignet.polynomial.legint(pol, m=j, k=list(range(j)), scl=2),
                    tolerance=1e-6,
                ),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legint(c2d, axis=0),
        numpy.vstack([beignet.polynomial.legint(c) for c in c2d.T]).T,
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legint(c2d, axis=1),
        numpy.vstack([beignet.polynomial.legint(c) for c in c2d]),
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legint(c2d, k=3, axis=1),
        numpy.vstack([beignet.polynomial.legint(c, k=3) for c in c2d]),
    )
    torch.testing.assert_close(beignet.polynomial.legint((1, 2, 3), 0), (1, 2, 3))
