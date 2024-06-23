import beignet.polynomial
import beignet.polynomial._evaluate_laguerre_series_1d
import beignet.polynomial._integrate_laguerre_series
import beignet.polynomial._laguerre_series_to_power_series
import beignet.polynomial._power_series_to_laguerre_series
import beignet.polynomial._trim_laguerre_series
import numpy


def test_integrate_laguerre_series():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagint.integrate_laguerre_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagint.integrate_laguerre_series, [0], -1
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagint.integrate_laguerre_series, [0], 1, [0, 0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagint.integrate_laguerre_series, [0], lbnd=[0]
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagint.integrate_laguerre_series, [0], scl=[0]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagint.integrate_laguerre_series, [0], axis=0.5
    )

    for i in range(2, 5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagint.integrate_laguerre_series(
                [0], m=i, k=([0] * (i - 2) + [1])
            ),
            [1, -1],
        )

    for i in range(5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagtrim.trim_laguerre_series(
                beignet.polynomial._lag2poly.laguerre_series_to_power_series(
                    beignet.polynomial._lagint.integrate_laguerre_series(
                        beignet.polynomial._poly2lag.power_series_to_laguerre_series(
                            [0] * i + [1]
                        ),
                        m=1,
                        k=[i],
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial._lagtrim.trim_laguerre_series(
                [i] + [0] * i + [1 / (i + 1)], tolerance=1e-6
            ),
        )

    for i in range(5):
        scl = i + 1
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.evaluate_laguerre_series_1d(
                -1,
                beignet.polynomial._lagint.integrate_laguerre_series(
                    beignet.polynomial._poly2lag.power_series_to_laguerre_series(
                        [0] * i + [1]
                    ),
                    m=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagtrim.trim_laguerre_series(
                beignet.polynomial._lag2poly.laguerre_series_to_power_series(
                    beignet.polynomial._lagint.integrate_laguerre_series(
                        beignet.polynomial._poly2lag.power_series_to_laguerre_series(
                            [0] * i + [1]
                        ),
                        m=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial._lagtrim.trim_laguerre_series(
                [i] + [0] * i + [2 / scl], tolerance=1e-6
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial._lagint.integrate_laguerre_series(tgt, m=1)
            res = beignet.polynomial._lagint.integrate_laguerre_series(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._lagint.integrate_laguerre_series(
                    tgt, m=1, k=[k]
                )
            res = beignet.polynomial._lagint.integrate_laguerre_series(
                pol, m=j, k=list(range(j))
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._lagint.integrate_laguerre_series(
                    tgt, m=1, k=[k], lbnd=-1
                )
            res = beignet.polynomial._lagint.integrate_laguerre_series(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial._lagint.integrate_laguerre_series(
                    tgt, m=1, k=[k], scl=2
                )
            res = beignet.polynomial._lagint.integrate_laguerre_series(
                pol, m=j, k=list(range(j)), scl=2
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagint.integrate_laguerre_series(c2d, axis=0),
        numpy.vstack(
            [beignet.polynomial._lagint.integrate_laguerre_series(c) for c in c2d.T]
        ).T,
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagint.integrate_laguerre_series(c2d, axis=1),
        numpy.vstack(
            [beignet.polynomial._lagint.integrate_laguerre_series(c) for c in c2d]
        ),
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagint.integrate_laguerre_series(c2d, k=3, axis=1),
        numpy.vstack(
            [beignet.polynomial._lagint.integrate_laguerre_series(c, k=3) for c in c2d]
        ),
    )
