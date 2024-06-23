import beignet.polynomial
import beignet.polynomial._evaluate_probabilists_hermite_series_1d
import beignet.polynomial._integrate_probabilists_hermite_series
import beignet.polynomial._power_series_to_probabilists_hermite_series
import beignet.polynomial._probabilists_hermite_series_to_power_series
import beignet.polynomial._trim_probabilists_hermite_series
import numpy


def test_hermeint():
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial._hermeint.integrate_probabilists_hermite_series,
        [0],
        0.5,
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._hermeint.integrate_probabilists_hermite_series,
        [0],
        -1,
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._hermeint.integrate_probabilists_hermite_series,
        [0],
        1,
        [0, 0],
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._hermeint.integrate_probabilists_hermite_series,
        [0],
        lbnd=[0],
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._hermeint.integrate_probabilists_hermite_series,
        [0],
        scl=[0],
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial._hermeint.integrate_probabilists_hermite_series,
        [0],
        axis=0.5,
    )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
            [0], m=i, k=k
        )
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermepol = (
            beignet.polynomial._poly2herme.power_series_to_probabilists_hermite_series(
                pol
            )
        )
        hermeint = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
            hermepol, m=1, k=[i]
        )
        res = (
            beignet.polynomial._herme2poly.probabilists_hermite_series_to_power_series(
                hermeint
            )
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                res, tolerance=1e-6
            ),
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                tgt, tolerance=1e-6
            ),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermepol = (
            beignet.polynomial._poly2herme.power_series_to_probabilists_hermite_series(
                pol
            )
        )
        hermeint = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
            hermepol, m=1, k=[i], lbnd=-1
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series_1d(
                -1, hermeint
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermepol = (
            beignet.polynomial._poly2herme.power_series_to_probabilists_hermite_series(
                pol
            )
        )
        hermeint = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
            hermepol, m=1, k=[i], scl=2
        )
        res = (
            beignet.polynomial._herme2poly.probabilists_hermite_series_to_power_series(
                hermeint
            )
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                res, tolerance=1e-6
            ),
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                tgt, tolerance=1e-6
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = (
                    beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                        tgt, m=1
                    )
                )
            res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                pol, m=j
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = (
                    beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                        tgt, m=1, k=[k]
                    )
                )
            res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                pol, m=j, k=list(range(j))
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = (
                    beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                        tgt, m=1, k=[k], lbnd=-1
                    )
                )
            res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                pol, m=j, k=list(range(j)), lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = (
                    beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                        tgt, m=1, k=[k], scl=2
                    )
                )
            res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                pol, m=j, k=list(range(j)), scl=2
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    tgt, tolerance=1e-6
                ),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack(
        [
            beignet.polynomial._hermeint.integrate_probabilists_hermite_series(c)
            for c in c2d.T
        ]
    ).T
    res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
        c2d, axis=0
    )
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack(
        [
            beignet.polynomial._hermeint.integrate_probabilists_hermite_series(c)
            for c in c2d
        ]
    )
    res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
        c2d, axis=1
    )
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack(
        [
            beignet.polynomial._hermeint.integrate_probabilists_hermite_series(c, k=3)
            for c in c2d
        ]
    )
    res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
        c2d, k=3, axis=1
    )
    numpy.testing.assert_almost_equal(res, tgt)
