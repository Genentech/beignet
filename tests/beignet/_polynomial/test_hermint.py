import beignet.polynomial
import beignet.polynomial._evaluate_1d_physicists_hermite_series
import beignet.polynomial._hermint
import beignet.polynomial._physicists_hermite_series_to_power_series
import beignet.polynomial._power_series_to_physicists_hermite_series
import beignet.polynomial._trim_physicists_hermite_series
import numpy


def test_hermint():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._hermint.hermint, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermint.hermint, [0], -1
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermint.hermint, [0], 1, [0, 0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermint.hermint, [0], lbnd=[0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermint.hermint, [0], scl=[0]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._hermint.hermint, [0], axis=0.5
    )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial._hermint.hermint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 0.5])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermpol = (
            beignet.polynomial._poly2herm.power_series_to_physicists_hermite_series(pol)
        )
        hermint = beignet.polynomial._hermint.hermint(hermpol, m=1, k=[i])
        res = beignet.polynomial._herm2poly.physicists_hermite_series_to_power_series(
            hermint
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                res, tolerance=1e-6
            ),
            beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                tgt, tolerance=1e-6
            ),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermpol = (
            beignet.polynomial._poly2herm.power_series_to_physicists_hermite_series(pol)
        )
        hermint = beignet.polynomial._hermint.hermint(hermpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermval.evaluate_1d_physicists_hermite_series(
                -1, hermint
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermpol = (
            beignet.polynomial._poly2herm.power_series_to_physicists_hermite_series(pol)
        )
        hermint = beignet.polynomial._hermint.hermint(hermpol, m=1, k=[i], scl=2)
        res = beignet.polynomial._herm2poly.physicists_hermite_series_to_power_series(
            hermint
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                res, tolerance=1e-6
            ),
            beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                tgt, tolerance=1e-6
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial._hermint.hermint(tgt, m=1)
            res = beignet.polynomial._hermint.hermint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._hermint.hermint(tgt, m=1, k=[k])
            res = beignet.polynomial._hermint.hermint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._hermint.hermint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial._hermint.hermint(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._hermint.hermint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial._hermint.hermint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial._hermint.hermint(c) for c in c2d.T]).T
    res = beignet.polynomial._hermint.hermint(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial._hermint.hermint(c) for c in c2d])
    res = beignet.polynomial._hermint.hermint(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial._hermint.hermint(c, k=3) for c in c2d])
    res = beignet.polynomial._hermint.hermint(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)
