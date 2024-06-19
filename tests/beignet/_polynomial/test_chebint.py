import beignet.polynomial
import beignet.polynomial._chebtrim
import numpy


def test_chebint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.chebint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        chebpol = beignet.polynomial.poly2cheb(pol)
        chebint = beignet.polynomial.chebint(chebpol, m=1, k=[i])
        res = beignet.polynomial.cheb2poly(chebint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
            beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        chebpol = beignet.polynomial.poly2cheb(pol)
        chebint = beignet.polynomial.chebint(chebpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.chebval(-1, chebint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        chebpol = beignet.polynomial.poly2cheb(pol)
        chebint = beignet.polynomial.chebint(chebpol, m=1, k=[i], scl=2)
        res = beignet.polynomial.cheb2poly(chebint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
            beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1)
            res = beignet.polynomial.chebint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1, k=[k])
            res = beignet.polynomial.chebint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.chebint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.chebint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.chebint(c) for c in c2d.T]).T
    res = beignet.polynomial.chebint(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.chebint(c) for c in c2d])
    res = beignet.polynomial.chebint(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.chebint(c, k=3) for c in c2d])
    res = beignet.polynomial.chebint(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)
