import beignet.polynomial
import numpy


def test_lagint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagint, [0], axis=0.5)

    for i in range(2, 5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagint([0], m=i, k=([0] * (i - 2) + [1])), [1, -1]
        )

    for i in range(5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagtrim(
                beignet.polynomial.lag2poly(
                    beignet.polynomial.lagint(
                        beignet.polynomial.poly2lag([0] * i + [1]), m=1, k=[i]
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.lagtrim([i] + [0] * i + [1 / (i + 1)], tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagval(
                -1,
                beignet.polynomial.lagint(
                    beignet.polynomial.poly2lag([0] * i + [1]), m=1, k=[i], lbnd=-1
                ),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagtrim(
                beignet.polynomial.lag2poly(
                    beignet.polynomial.lagint(
                        beignet.polynomial.poly2lag([0] * i + [1]), m=1, k=[i], scl=2
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.lagtrim([i] + [0] * i + [2 / scl], tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1)
            res = beignet.polynomial.lagint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1, k=[k])
            res = beignet.polynomial.lagint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.lagint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagint(c2d, axis=0),
        numpy.vstack([beignet.polynomial.lagint(c) for c in c2d.T]).T,
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagint(c2d, axis=1),
        numpy.vstack([beignet.polynomial.lagint(c) for c in c2d]),
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagint(c2d, k=3, axis=1),
        numpy.vstack([beignet.polynomial.lagint(c, k=3) for c in c2d]),
    )
