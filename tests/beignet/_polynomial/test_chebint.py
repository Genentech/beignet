import beignet.polynomial
import beignet.polynomial._chebyshev_series_to_power_series
import beignet.polynomial._evaluate_chebyshev_series_1d
import beignet.polynomial._integrate_chebyshev_series
import beignet.polynomial._power_series_to_chebyshev_series
import beignet.polynomial._trim_chebyshev_series
import numpy


def test_chebint():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._chebint.integrate_chebyshev_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._chebint.integrate_chebyshev_series, [0], -1
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._chebint.integrate_chebyshev_series,
        [0],
        1,
        [0, 0],
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._chebint.integrate_chebyshev_series,
        [0],
        lbnd=[0],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._chebint.integrate_chebyshev_series, [0], scl=[0]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._chebint.integrate_chebyshev_series, [0], axis=0.5
    )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial._chebint.integrate_chebyshev_series([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        chebpol = beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(pol)
        chebint = beignet.polynomial._chebint.integrate_chebyshev_series(
            chebpol, m=1, k=[i]
        )
        res = beignet.polynomial._cheb2poly.chebyshev_series_to_power_series(chebint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.trim_chebyshev_series(res, tolerance=1e-6),
            beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        chebpol = beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(pol)
        chebint = beignet.polynomial._chebint.integrate_chebyshev_series(
            chebpol, m=1, k=[i], lbnd=-1
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series_1d(-1, chebint), i
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        chebpol = beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(pol)
        chebint = beignet.polynomial._chebint.integrate_chebyshev_series(
            chebpol, m=1, k=[i], scl=2
        )
        res = beignet.polynomial._cheb2poly.chebyshev_series_to_power_series(chebint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.trim_chebyshev_series(res, tolerance=1e-6),
            beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial._chebint.integrate_chebyshev_series(tgt, m=1)
            res = beignet.polynomial._chebint.integrate_chebyshev_series(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.trim_chebyshev_series(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._chebint.integrate_chebyshev_series(
                    tgt, m=1, k=[k]
                )
            res = beignet.polynomial._chebint.integrate_chebyshev_series(
                pol, m=j, k=list(range(j))
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.trim_chebyshev_series(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._chebint.integrate_chebyshev_series(
                    tgt, m=1, k=[k], lbnd=-1
                )
            res = beignet.polynomial._chebint.integrate_chebyshev_series(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.trim_chebyshev_series(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._chebint.integrate_chebyshev_series(
                    tgt, m=1, k=[k], scl=2
                )
            res = beignet.polynomial._chebint.integrate_chebyshev_series(
                pol, m=j, k=list(range(j)), scl=2
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.trim_chebyshev_series(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack(
        [beignet.polynomial._chebint.integrate_chebyshev_series(c) for c in c2d.T]
    ).T
    res = beignet.polynomial._chebint.integrate_chebyshev_series(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack(
        [beignet.polynomial._chebint.integrate_chebyshev_series(c) for c in c2d]
    )
    res = beignet.polynomial._chebint.integrate_chebyshev_series(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack(
        [beignet.polynomial._chebint.integrate_chebyshev_series(c, k=3) for c in c2d]
    )
    res = beignet.polynomial._chebint.integrate_chebyshev_series(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)
