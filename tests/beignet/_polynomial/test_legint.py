import beignet.polynomial
import beignet.polynomial._evaluate_legendre_series_1d
import beignet.polynomial._integrate_legendre_series
import beignet.polynomial._legendre_series_to_power_series
import beignet.polynomial._power_series_to_legendre_series
import beignet.polynomial._trim_legendre_series
import numpy
import torch


def test_integrate_legendre_series():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._legint.integrate_legendre_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legint.integrate_legendre_series, [0], -1
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legint.integrate_legendre_series, [0], 1, [0, 0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legint.integrate_legendre_series, [0], lbnd=[0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legint.integrate_legendre_series, [0], scl=[0]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._legint.integrate_legendre_series, [0], axis=0.5
    )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legint.integrate_legendre_series([0], m=i, k=k), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        legpol = beignet.polynomial._poly2leg.power_series_to_legendre_series(pol)
        legint = beignet.polynomial._legint.integrate_legendre_series(
            legpol, m=1, k=[i]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legtrim.trim_legendre_series(
                beignet.polynomial._leg2poly.legendre_series_to_power_series(legint),
                tolerance=1e-6,
            ),
            beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        legpol = beignet.polynomial._poly2leg.power_series_to_legendre_series(pol)
        legint = beignet.polynomial._legint.integrate_legendre_series(
            legpol, m=1, k=[i], lbnd=-1
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.evaluate_legendre_series_1d(-1, legint), i
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        legpol = beignet.polynomial._poly2leg.power_series_to_legendre_series(pol)
        legint = beignet.polynomial._legint.integrate_legendre_series(
            legpol, m=1, k=[i], scl=2
        )
        res = beignet.polynomial._leg2poly.legendre_series_to_power_series(legint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6),
            beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial._legint.integrate_legendre_series(tgt, m=1)
            res = beignet.polynomial._legint.integrate_legendre_series(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6),
                beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._legint.integrate_legendre_series(
                    tgt, m=1, k=[k]
                )
            res = beignet.polynomial._legint.integrate_legendre_series(
                pol, m=j, k=list(range(j))
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6),
                beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._legint.integrate_legendre_series(
                    tgt, m=1, k=[k], lbnd=-1
                )
            res = beignet.polynomial._legint.integrate_legendre_series(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6),
                beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._legint.integrate_legendre_series(
                    tgt, m=1, k=[k], scl=2
                )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legtrim.trim_legendre_series(
                    beignet.polynomial._legint.integrate_legendre_series(
                        pol, m=j, k=list(range(j)), scl=2
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legint.integrate_legendre_series(c2d, axis=0),
        numpy.vstack(
            [beignet.polynomial._legint.integrate_legendre_series(c) for c in c2d.T]
        ).T,
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legint.integrate_legendre_series(c2d, axis=1),
        numpy.vstack(
            [beignet.polynomial._legint.integrate_legendre_series(c) for c in c2d]
        ),
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legint.integrate_legendre_series(c2d, k=3, axis=1),
        numpy.vstack(
            [beignet.polynomial._legint.integrate_legendre_series(c, k=3) for c in c2d]
        ),
    )
    torch.testing.assert_close(
        beignet.polynomial._legint.integrate_legendre_series((1, 2, 3), 0), (1, 2, 3)
    )
