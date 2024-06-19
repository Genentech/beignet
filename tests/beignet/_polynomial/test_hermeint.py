import beignet.polynomial
import numpy


def test_hermeint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermeint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermeint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.hermeint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermepol = beignet.polynomial.poly2herme(pol)
        hermeint = beignet.polynomial.hermeint(hermepol, m=1, k=[i])
        res = beignet.polynomial.herme2poly(hermeint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermetrim(res, tolerance=1e-6),
            beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermepol = beignet.polynomial.poly2herme(pol)
        hermeint = beignet.polynomial.hermeint(hermepol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(-1, hermeint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermepol = beignet.polynomial.poly2herme(pol)
        hermeint = beignet.polynomial.hermeint(hermepol, m=1, k=[i], scl=2)
        res = beignet.polynomial.herme2poly(hermeint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermetrim(res, tolerance=1e-6),
            beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1)
            res = beignet.polynomial.hermeint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1, k=[k])
            res = beignet.polynomial.hermeint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.hermeint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.hermeint(c) for c in c2d.T]).T
    res = beignet.polynomial.hermeint(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermeint(c) for c in c2d])
    res = beignet.polynomial.hermeint(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermeint(c, k=3) for c in c2d])
    res = beignet.polynomial.hermeint(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)
