import beignet.polynomial
import numpy
import torch


def test_integrate_legendre_series():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.integrate_legendre_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_legendre_series, [0], -1
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_legendre_series, [0], 1, [0, 0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_legendre_series, [0], lbnd=[0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_legendre_series, [0], scl=[0]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._legint.integrate_legendre_series, [0], axis=0.5
    )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        torch.testing.assert_close(
            beignet.polynomial.integrate_legendre_series([0], m=i, k=k), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        legpol = beignet.polynomial.power_series_to_legendre_series(pol)
        legint = beignet.polynomial.integrate_legendre_series(legpol, m=1, k=[i])
        torch.testing.assert_close(
            beignet.polynomial.trim_legendre_series(
                beignet.polynomial.legendre_series_to_power_series(legint),
                tolerance=0.000001,
            ),
            beignet.polynomial.trim_legendre_series(tgt, tolerance=0.000001),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        legpol = beignet.polynomial.power_series_to_legendre_series(pol)
        legint = beignet.polynomial.integrate_legendre_series(
            legpol, m=1, k=[i], lbnd=-1
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_legendre_series_1d(-1, legint), i
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        legpol = beignet.polynomial.power_series_to_legendre_series(pol)
        legint = beignet.polynomial.integrate_legendre_series(legpol, m=1, k=[i], scl=2)
        res = beignet.polynomial.legendre_series_to_power_series(legint)
        torch.testing.assert_close(
            beignet.polynomial.trim_legendre_series(res, tolerance=0.000001),
            beignet.polynomial.trim_legendre_series(tgt, tolerance=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.integrate_legendre_series(tgt, m=1)
            res = beignet.polynomial.integrate_legendre_series(pol, m=j)
            torch.testing.assert_close(
                beignet.polynomial.trim_legendre_series(res, tolerance=0.000001),
                beignet.polynomial.trim_legendre_series(tgt, tolerance=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.integrate_legendre_series(tgt, m=1, k=[k])
            res = beignet.polynomial.integrate_legendre_series(
                pol, m=j, k=list(range(j))
            )
            torch.testing.assert_close(
                beignet.polynomial.trim_legendre_series(res, tolerance=0.000001),
                beignet.polynomial.trim_legendre_series(tgt, tolerance=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.integrate_legendre_series(
                    tgt, m=1, k=[k], lbnd=-1
                )
            res = beignet.polynomial.integrate_legendre_series(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            torch.testing.assert_close(
                beignet.polynomial.trim_legendre_series(res, tolerance=0.000001),
                beignet.polynomial.trim_legendre_series(tgt, tolerance=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.integrate_legendre_series(
                    tgt, m=1, k=[k], scl=2
                )
            torch.testing.assert_close(
                beignet.polynomial.trim_legendre_series(
                    beignet.polynomial.integrate_legendre_series(
                        pol, m=j, k=list(range(j)), scl=2
                    ),
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_legendre_series(tgt, tolerance=0.000001),
            )

    c2d = numpy.random.random((3, 4))
    torch.testing.assert_close(
        beignet.polynomial.integrate_legendre_series(c2d, axis=0),
        numpy.vstack(
            [beignet.polynomial.integrate_legendre_series(c) for c in c2d.T]
        ).T,
    )
    torch.testing.assert_close(
        beignet.polynomial.integrate_legendre_series(c2d, axis=1),
        numpy.vstack([beignet.polynomial.integrate_legendre_series(c) for c in c2d]),
    )
    torch.testing.assert_close(
        beignet.polynomial.integrate_legendre_series(c2d, k=3, axis=1),
        numpy.vstack(
            [beignet.polynomial.integrate_legendre_series(c, k=3) for c in c2d]
        ),
    )
    torch.testing.assert_close(
        beignet.polynomial.integrate_legendre_series((1, 2, 3), 0), (1, 2, 3)
    )
