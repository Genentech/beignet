import functools

import beignet._add_probabilists_hermite_series
import beignet._differentiate_probabilists_hermite_series
import beignet._divide_probabilists_hermite_series
import beignet._evaluate_2d_probabilists_hermite_series
import beignet._evaluate_3d_probabilists_hermite_series
import beignet._evaluate_power_series
import beignet._evaluate_probabilists_hermite_series
import beignet._fit_probabilists_hermite_series
import beignet._integrate_probabilists_hermite_series
import beignet._multiply_probabilists_hermite_series
import beignet._power_series_to_probabilists_hermite_series
import beignet._probabilists_hermite_series_from_roots
import beignet._probabilists_hermite_series_roots
import beignet._probabilists_hermite_series_to_power_series
import beignet._subtract_probabilists_hermite_series
import beignet._trim_probabilists_hermite_series
import beignet.polynomial
import beignet.polynomial._hermecompanion
import beignet.polynomial._hermedomain
import beignet.polynomial._hermegauss
import beignet.polynomial._hermegrid2d
import beignet.polynomial._hermegrid3d
import beignet.polynomial._hermeline
import beignet.polynomial._hermemulx
import beignet.polynomial._hermeone
import beignet.polynomial._hermepow
import beignet.polynomial._hermevander
import beignet.polynomial._hermevander2d
import beignet.polynomial._hermevander3d
import beignet.polynomial._hermeweight
import beignet.polynomial._hermex
import beignet.polynomial._hermezero
import numpy
import numpy.testing

He0 = numpy.array([1])
He1 = numpy.array([0, 1])
He2 = numpy.array([-1, 0, 1])
He3 = numpy.array([0, -3, 0, 1])
He4 = numpy.array([3, 0, -6, 0, 1])
He5 = numpy.array([0, 15, 0, -10, 0, 1])
He6 = numpy.array([-15, 0, 45, 0, -15, 0, 1])
He7 = numpy.array([0, -105, 0, 105, 0, -21, 0, 1])
He8 = numpy.array([105, 0, -420, 0, 210, 0, -28, 0, 1])
He9 = numpy.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])

Helist = [He0, He1, He2, He3, He4, He5, He6, He7, He8, He9]


def trim(x):
    return beignet.polynomial._hermetrim.trim_probabilists_hermite_series(x, tol=1e-6)


class TestConstants:
    def test_hermedomain(self):
        numpy.testing.assert_equal(beignet.polynomial._hermedomain.hermedomain, [-1, 1])

    def test_hermezero(self):
        numpy.testing.assert_equal(beignet.polynomial._hermezero.hermezero, [0])

    def test_hermeone(self):
        numpy.testing.assert_equal(beignet.polynomial._hermeone.hermeone, [1])

    def test_hermex(self):
        numpy.testing.assert_equal(beignet.polynomial._hermex.hermex, [0, 1])


class TestArithmetic:
    x = numpy.linspace(-3, 3, 100)

    def test_hermeadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.polynomial._hermeadd.add_probabilists_hermite_series(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermesub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.polynomial._hermesub.subtract_probabilists_hermite_series(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermemulx(self):
        numpy.testing.assert_equal(beignet.polynomial._hermemulx.hermemulx([0]), [0])
        numpy.testing.assert_equal(beignet.polynomial._hermemulx.hermemulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i, 0, 1]
            numpy.testing.assert_equal(
                beignet.polynomial._hermemulx.hermemulx(ser), tgt
            )

    def test_hermemul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                self.x, pol1
            )
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = (
                    beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                        self.x, pol2
                    )
                )
                pol3 = (
                    beignet.polynomial._hermemul.multiply_probabilists_hermite_series(
                        pol1, pol2
                    )
                )
                val3 = (
                    beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                        self.x, pol3
                    )
                )
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_hermediv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.polynomial._hermeadd.add_probabilists_hermite_series(
                    ci, cj
                )
                quo, rem = (
                    beignet.polynomial._hermediv.divide_probabilists_hermite_series(
                        tgt, ci
                    )
                )
                res = beignet.polynomial._hermeadd.add_probabilists_hermite_series(
                    beignet.polynomial._hermemul.multiply_probabilists_hermite_series(
                        quo, ci
                    ),
                    rem,
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermepow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.polynomial._hermemul.multiply_probabilists_hermite_series,
                    [c] * j,
                    numpy.array([1]),
                )
                res = beignet.polynomial._hermepow.hermepow(c, j)
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.evaluate_power_series(x, [1.0, 2.0, 3.0])

    def test_hermeval(self):
        # check empty input
        numpy.testing.assert_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                [], [1]
            ).size,
            0,
        )

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [beignet.polynomial._polyval.evaluate_power_series(x, c) for c in Helist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                x, [0] * i + [1]
            )
            numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                    x, [1]
                ).shape,
                dims,
            )
            numpy.testing.assert_equal(
                beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                    x, [1, 0]
                ).shape,
                dims,
            )
            numpy.testing.assert_equal(
                beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                    x, [1, 0, 0]
                ).shape,
                dims,
            )

    def test_hermeval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermeval2d.evaluate_2d_probabilists_hermite_series,
            x1,
            x2[:2],
            self.c2d,
        )

        # test values
        tgt = y1 * y2
        res = beignet.polynomial._hermeval2d.evaluate_2d_probabilists_hermite_series(
            x1, x2, self.c2d
        )
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._hermeval2d.evaluate_2d_probabilists_hermite_series(
            z, z, self.c2d
        )
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermeval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermeval3d.evaluate_3d_probabilists_hermite_series,
            x1,
            x2,
            x3[:2],
            self.c3d,
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.polynomial._hermeval3d.evaluate_3d_probabilists_hermite_series(
            x1, x2, x3, self.c3d
        )
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._hermeval3d.evaluate_3d_probabilists_hermite_series(
            z, z, z, self.c3d
        )
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermegrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.polynomial._hermegrid2d.hermegrid2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._hermegrid2d.hermegrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_hermegrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.polynomial._hermegrid3d.hermegrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._hermegrid3d.hermegrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_hermeint(self):
        # check exceptions
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

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                [0], m=i, k=k
            )
            numpy.testing.assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermepol = beignet.power_series_to_probabilists_hermite_series(pol)
            hermeint = beignet.integrate_probabilists_hermite_series(
                hermepol, m=1, k=[i]
            )
            res = beignet.herme_to_power_series(hermeint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            hermepol = beignet.power_series_to_probabilists_hermite_series(pol)
            hermeint = beignet.integrate_probabilists_hermite_series(
                hermepol, m=1, k=[i], lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.evaluate_probabilists_hermite_series(-1, hermeint),
                i,
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            hermepol = beignet.power_series_to_probabilists_hermite_series(pol)
            hermeint = beignet.integrate_probabilists_hermite_series(
                hermepol, m=1, k=[i], scl=2
            )
            res = beignet.herme_to_power_series(hermeint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.integrate_probabilists_hermite_series(tgt, m=1)
                res = beignet.integrate_probabilists_hermite_series(pol, m=j)
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.integrate_probabilists_hermite_series(tgt, m=1, k=[k])
                res = beignet.integrate_probabilists_hermite_series(
                    pol, m=j, k=list(range(j))
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.integrate_probabilists_hermite_series(
                        tgt, m=1, k=[k], lbnd=-1
                    )
                res = beignet.integrate_probabilists_hermite_series(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.integrate_probabilists_hermite_series(
                        tgt, m=1, k=[k], scl=2
                    )
                res = beignet.integrate_probabilists_hermite_series(
                    pol, m=j, k=list(range(j)), scl=2
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_hermeint_axis(self):
        # check that axis keyword works
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
                beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
                    c, k=3
                )
                for c in c2d
            ]
        )
        res = beignet.polynomial._hermeint.integrate_probabilists_hermite_series(
            c2d, k=3, axis=1
        )
        numpy.testing.assert_almost_equal(res, tgt)


class TestDerivative:
    def test_hermeder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermeder.differentiate_probabilists_hermite_series,
            [0],
            0.5,
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermeder.differentiate_probabilists_hermite_series,
            [0],
            -1,
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = (
                beignet.polynomial._hermeder.differentiate_probabilists_hermite_series(
                    tgt, m=0
                )
            )
            numpy.testing.assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.differentiate_probabilists_hermite_series(
                    beignet.integrate_probabilists_hermite_series(tgt, m=j),
                    m=j,
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.differentiate_probabilists_hermite_series(
                    beignet.integrate_probabilists_hermite_series(tgt, m=j, scl=2),
                    m=j,
                    scl=0.5,
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_hermeder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack(
            [
                beignet.polynomial._hermeder.differentiate_probabilists_hermite_series(
                    c
                )
                for c in c2d.T
            ]
        ).T
        res = beignet.polynomial._hermeder.differentiate_probabilists_hermite_series(
            c2d, axis=0
        )
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack(
            [
                beignet.polynomial._hermeder.differentiate_probabilists_hermite_series(
                    c
                )
                for c in c2d
            ]
        )
        res = beignet.polynomial._hermeder.differentiate_probabilists_hermite_series(
            c2d, axis=1
        )
        numpy.testing.assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_hermevander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.polynomial._hermevander.hermevander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i],
                beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                    x, coef
                ),
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.polynomial._hermevander.hermevander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i],
                beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(
                    x, coef
                ),
            )

    def test_hermevander2d(self):
        # also tests hermeval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.polynomial._hermevander2d.hermevander2d(x1, x2, [1, 2])
        tgt = beignet.polynomial._hermeval2d.evaluate_2d_probabilists_hermite_series(
            x1, x2, c
        )
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._hermevander2d.hermevander2d([x1], [x2], [1, 2])
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_hermevander3d(self):
        # also tests hermeval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.polynomial._hermevander3d.hermevander3d(x1, x2, x3, [1, 2, 3])
        tgt = beignet.polynomial._hermeval3d.evaluate_3d_probabilists_hermite_series(
            x1, x2, x3, c
        )
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._hermevander3d.hermevander3d(
            [x1], [x2], [x3], [1, 2, 3]
        )
        numpy.testing.assert_(van.shape == (1, 5, 24))


class TestFitting:
    def test_hermefit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1],
            -1,
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [[1]],
            [1],
            0,
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [],
            [1],
            0,
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [[[1]]],
            0,
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1, 2],
            [1],
            0,
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1, 2],
            0,
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1],
            0,
            w=[[1]],
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1],
            0,
            w=[1, 1],
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1],
            [
                -1,
            ],
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1],
            [2, -1, 6],
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._hermefit.fit_probabilists_hermite_series,
            [1],
            [1],
            [],
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef3),
            y,
        )
        coef3 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, y, [0, 1, 2, 3]
        )
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef3),
            y,
        )
        #
        coef4 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef4),
            y,
        )
        coef4 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, y, [0, 1, 2, 3, 4]
        )
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef4),
            y,
        )
        # check things still work if deg is not in strict increasing
        coef4 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, y, [2, 3, 4, 1, 0]
        )
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef4),
            y,
        )
        #
        coef2d = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, numpy.array([y, y]).T, 3
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, numpy.array([y, y]).T, [0, 1, 2, 3]
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, yw, 3, w=w
        )
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, yw, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, numpy.array([yw, yw]).T, 3, w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermefit.fit_probabilists_hermite_series(x, x, 1),
            [0, 1],
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermefit.fit_probabilists_hermite_series(x, x, [0, 1]),
            [0, 1],
        )
        # test fitting only even Legendre polynomials
        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(x, y, 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef1),
            y,
        )
        coef2 = beignet.polynomial._hermefit.fit_probabilists_hermite_series(
            x, y, [0, 2, 4]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeval.evaluate_probabilists_hermite_series(x, coef2),
            y,
        )
        numpy.testing.assert_almost_equal(coef1, coef2)


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._hermecompanion.hermecompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._hermecompanion.hermecompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.polynomial._hermecompanion.hermecompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.polynomial._hermecompanion.hermecompanion([1, 2])[0, 0] == -0.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.polynomial._hermegauss.hermegauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = beignet.polynomial._hermevander.hermevander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_almost_equal(vv, numpy.eye(100))

        # check that the integral of 1 is correct
        tgt = numpy.sqrt(2 * numpy.pi)
        numpy.testing.assert_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_hermefromroots(self):
        res = beignet.polynomial._hermefromroots.probabilists_hermite_series_from_roots(
            []
        )
        numpy.testing.assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            pol = beignet.probabilists_hermite_series_from_roots(roots)
            res = beignet.evaluate_probabilists_hermite_series(roots, pol)
            tgt = 0
            numpy.testing.assert_(len(pol) == i + 1)
            numpy.testing.assert_almost_equal(beignet.herme_to_power_series(pol)[-1], 1)
            numpy.testing.assert_almost_equal(res, tgt)

    def test_hermeroots(self):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeroots.probabilists_hermite_series_roots([1]), []
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermeroots.probabilists_hermite_series_roots([1, 1]),
            [-1],
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.polynomial._hermeroots.probabilists_hermite_series_roots(
                beignet.polynomial._hermefromroots.probabilists_hermite_series_from_roots(
                    tgt
                )
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_hermetrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series,
            coef,
            -1,
        )

        # Test results
        numpy.testing.assert_equal(
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(coef),
            coef[:-1],
        )
        numpy.testing.assert_equal(
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(coef, 1),
            coef[:-3],
        )
        numpy.testing.assert_equal(
            beignet.polynomial._hermetrim.trim_probabilists_hermite_series(coef, 2), [0]
        )

    def test_hermeline(self):
        numpy.testing.assert_equal(
            beignet.polynomial._hermeline.hermeline(3, 4), [3, 4]
        )

    def test_herme2poly(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._herme2poly.herme_to_power_series([0] * i + [1]),
                Helist[i],
            )

    def test_poly2herme(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._poly2herme.power_series_to_probabilists_hermite_series(
                    Helist[i]
                ),
                [0] * i + [1],
            )

    def test_weight(self):
        x = numpy.linspace(-5, 5, 11)
        tgt = numpy.exp(-0.5 * x**2)
        res = beignet.polynomial._hermeweight.hermeweight(x)
        numpy.testing.assert_almost_equal(res, tgt)
