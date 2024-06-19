import beignet.polynomial
import numpy


def test_polyint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.polyint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polyint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polyint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polyint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polyint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.polyint, [0], axis=0.5)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.polyint, [1, 1], 1.0)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.polyint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        res = beignet.polynomial.polyint(pol, m=1, k=[i])
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polytrim(res, tolerance=1e-6),
            beignet.polynomial.polytrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        res = beignet.polynomial.polyint(pol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.polyval(-1, res), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        res = beignet.polynomial.polyint(pol, m=1, k=[i], scl=2)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polytrim(res, tolerance=1e-6),
            beignet.polynomial.polytrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.polyint(tgt, m=1)
            res = beignet.polynomial.polyint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.polyint(tgt, m=1, k=[k])
            res = beignet.polynomial.polyint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.polyint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.polyint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.polyint(tgt, m=1, k=[k], scl=2)

            numpy.testing.assert_almost_equal(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polyint(
                        pol,
                        m=j,
                        k=list(range(j)),
                        scl=2,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    numpy.testing.assert_almost_equal(
        beignet.polynomial.polyint(c2d, axis=0),
        numpy.vstack([beignet.polynomial.polyint(c) for c in c2d.T]).T,
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.polyint(c2d, axis=1),
        numpy.vstack([beignet.polynomial.polyint(c) for c in c2d]),
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.polyint(c2d, k=3, axis=1),
        numpy.vstack([beignet.polynomial.polyint(c, k=3) for c in c2d]),
    )
