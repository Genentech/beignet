import beignet.polynomial
import numpy
import torch


def test_integrate_power_series():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.integrate_power_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_power_series, [0], -1
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_power_series, [0], 1, [0, 0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_power_series, [0], lbnd=[0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.integrate_power_series, [0], scl=[0]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.integrate_power_series, [0], axis=0.5
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.integrate_power_series, [1, 1], 1.0
    )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.integrate_power_series([0], m=i, k=k)
        torch.testing.assert_close(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        res = beignet.polynomial.integrate_power_series(pol, m=1, k=[i])
        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(res, tolerance=1e-6),
            beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        res = beignet.polynomial.integrate_power_series(pol, m=1, k=[i], lbnd=-1)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d(-1, res), i
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        res = beignet.polynomial.integrate_power_series(pol, m=1, k=[i], scl=2)
        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(res, tolerance=1e-6),
            beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.integrate_power_series(tgt, m=1)
            res = beignet.polynomial.integrate_power_series(pol, m=j)
            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.integrate_power_series(tgt, m=1, k=[k])
            res = beignet.polynomial.integrate_power_series(pol, m=j, k=list(range(j)))
            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.integrate_power_series(
                    tgt, m=1, k=[k], lbnd=-1
                )
            res = beignet.polynomial.integrate_power_series(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.integrate_power_series(tgt, m=1, k=[k], scl=2)

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(
                    beignet.polynomial.integrate_power_series(
                        pol,
                        m=j,
                        k=list(range(j)),
                        scl=2,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    torch.testing.assert_close(
        beignet.polynomial.integrate_power_series(c2d, axis=0),
        numpy.vstack([beignet.polynomial.integrate_power_series(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.integrate_power_series(c2d, axis=1),
        numpy.vstack([beignet.polynomial.integrate_power_series(c) for c in c2d]),
    )

    torch.testing.assert_close(
        beignet.polynomial.integrate_power_series(c2d, k=3, axis=1),
        numpy.vstack([beignet.polynomial.integrate_power_series(c, k=3) for c in c2d]),
    )
