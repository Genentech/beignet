import functools

import beignet._add_chebyshev_series
import beignet._chebyshev_gauss_quadrature
import beignet._chebyshev_series_from_roots
import beignet._chebyshev_series_roots
import beignet._chebyshev_series_to_polynomial_series
import beignet._chebyshev_series_vandermonde
import beignet._differentiate_chebyshev_series
import beignet._divide_chebyshev_series
import beignet._evaluate_2d_chebyshev_series
import beignet._evaluate_3d_chebyshev_series
import beignet._evaluate_chebyshev_series
import beignet._evaluate_power_series
import beignet._fit_chebyshev_series
import beignet._integrate_chebyshev_series
import beignet._multiply_chebyshev_series
import beignet._power_series_to_chebyshev_series
import beignet._subtract_chebyshev_series
import beignet._trim_chebyshev_series
import beignet.polynomial
import beignet.polynomial.__cseries_to_zseries
import beignet.polynomial.__zseries_to_cseries
import beignet.polynomial._chebcompanion
import beignet.polynomial._chebdomain
import beignet.polynomial._chebgrid2d
import beignet.polynomial._chebgrid3d
import beignet.polynomial._chebinterpolate
import beignet.polynomial._chebline
import beignet.polynomial._chebmulx
import beignet.polynomial._chebone
import beignet.polynomial._chebpow
import beignet.polynomial._chebpts1
import beignet.polynomial._chebpts2
import beignet.polynomial._chebvander2d
import beignet.polynomial._chebvander3d
import beignet.polynomial._chebweight
import beignet.polynomial._chebx
import beignet.polynomial._chebzero
import numpy
import numpy.testing


def trim(x):
    return beignet.polynomial._chebtrim.trim_chebyshev_series(x, tol=1e-6)


T0 = [1]
T1 = [0, 1]
T2 = [-1, 0, 2]
T3 = [0, -3, 0, 4]
T4 = [1, 0, -8, 0, 8]
T5 = [0, 5, 0, -20, 0, 16]
T6 = [-1, 0, 18, 0, -48, 0, 32]
T7 = [0, -7, 0, 56, 0, -112, 0, 64]
T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]


class TestPrivate:
    def test__cseries_to_zseries(self):
        for i in range(5):
            inp = numpy.array([2] + [1] * i, numpy.double)
            tgt = numpy.array([0.5] * i + [2] + [0.5] * i, numpy.double)
            res = beignet.polynomial._cseries_to_zseries(inp)
            numpy.testing.assert_equal(res, tgt)

    def test__zseries_to_cseries(self):
        for i in range(5):
            inp = numpy.array([0.5] * i + [2] + [0.5] * i, numpy.double)
            tgt = numpy.array([2] + [1] * i, numpy.double)
            res = beignet.polynomial._zseries_to_cseries(inp)
            numpy.testing.assert_equal(res, tgt)


class TestConstants:
    def test_chebdomain(self):
        numpy.testing.assert_equal(beignet.polynomial._chebdomain.chebdomain, [-1, 1])

    def test_chebzero(self):
        numpy.testing.assert_equal(beignet.polynomial._chebzero.chebzero, [0])

    def test_chebone(self):
        numpy.testing.assert_equal(beignet.polynomial._chebone.chebone, [1])

    def test_chebx(self):
        numpy.testing.assert_equal(beignet.polynomial._chebx.chebx, [0, 1])


class TestArithmetic:
    def test_chebadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.polynomial.add_chebyshev_polynomial(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.polynomial._chebsub.subtract_chebyshev_series(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebmulx(self):
        numpy.testing.assert_equal(beignet.polynomial._chebmulx.chebmulx([0]), [0])
        numpy.testing.assert_equal(beignet.polynomial._chebmulx.chebmulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [0.5, 0, 0.5]
            numpy.testing.assert_equal(beignet.polynomial._chebmulx.chebmulx(ser), tgt)

    def test_chebmul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(i + j + 1)
                tgt[i + j] += 0.5
                tgt[abs(i - j)] += 0.5
                res = beignet.polynomial._chebmul.multiply_chebyshev_series(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.polynomial.add_chebyshev_polynomial(ci, cj)
                quo, rem = beignet.polynomial._chebdiv.divide_chebyshev_series(tgt, ci)
                res = beignet.polynomial.add_chebyshev_polynomial(
                    beignet.polynomial._chebmul.multiply_chebyshev_series(quo, ci), rem
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.polynomial._chebmul.multiply_chebyshev_series,
                    [c] * j,
                    numpy.array([1]),
                )
                res = beignet.polynomial._chebpow.chebpow(c, j)
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.evaluate_power_series(x, [1.0, 2.0, 3.0])

    def test_chebval(self):
        # check empty input
        numpy.testing.assert_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series([], [1]).size, 0
        )

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [beignet.polynomial._polyval.evaluate_power_series(x, c) for c in Tlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.polynomial._chebval.evaluate_chebyshev_series(
                x, [0] * i + [1]
            )
            numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial._chebval.evaluate_chebyshev_series(x, [1]).shape,
                dims,
            )
            numpy.testing.assert_equal(
                beignet.polynomial._chebval.evaluate_chebyshev_series(x, [1, 0]).shape,
                dims,
            )
            numpy.testing.assert_equal(
                beignet.polynomial._chebval.evaluate_chebyshev_series(
                    x, [1, 0, 0]
                ).shape,
                dims,
            )

    def test_chebval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._chebval2d.evaluate_2d_chebyshev_series,
            x1,
            x2[:2],
            self.c2d,
        )

        # test values
        tgt = y1 * y2
        res = beignet.polynomial._chebval2d.evaluate_2d_chebyshev_series(
            x1, x2, self.c2d
        )
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._chebval2d.evaluate_2d_chebyshev_series(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_chebval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._chebval3d.evaluate_3d_chebyshev_series,
            x1,
            x2,
            x3[:2],
            self.c3d,
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.polynomial._chebval3d.evaluate_3d_chebyshev_series(
            x1, x2, x3, self.c3d
        )
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._chebval3d.evaluate_3d_chebyshev_series(
            z, z, z, self.c3d
        )
        numpy.testing.assert_(res.shape == (2, 3))

    def test_chebgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.polynomial._chebgrid2d.chebgrid2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._chebgrid2d.chebgrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_chebgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.polynomial._chebgrid3d.chebgrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._chebgrid3d.chebgrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_chebint(self):
        # check exceptions
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
            ValueError,
            beignet.polynomial._chebint.integrate_chebyshev_series,
            [0],
            scl=[0],
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._chebint.integrate_chebyshev_series,
            [0],
            axis=0.5,
        )

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.polynomial._chebint.integrate_chebyshev_series([0], m=i, k=k)
            numpy.testing.assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            chebpol = beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(
                pol
            )
            chebint = beignet.polynomial._chebint.integrate_chebyshev_series(
                chebpol, m=1, k=[i]
            )
            res = beignet.polynomial._cheb2poly.chebyshev_series_to_polynomial_series(
                chebint
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            chebpol = beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(
                pol
            )
            chebint = beignet.polynomial._chebint.integrate_chebyshev_series(
                chebpol, m=1, k=[i], lbnd=-1
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebval.evaluate_chebyshev_series(-1, chebint), i
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            chebpol = beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(
                pol
            )
            chebint = beignet.polynomial._chebint.integrate_chebyshev_series(
                chebpol, m=1, k=[i], scl=2
            )
            res = beignet.polynomial._cheb2poly.chebyshev_series_to_polynomial_series(
                chebint
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.polynomial._chebint.integrate_chebyshev_series(
                        tgt, m=1
                    )
                res = beignet.polynomial._chebint.integrate_chebyshev_series(pol, m=j)
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
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
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
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
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
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
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_chebint_axis(self):
        # check that axis keyword works
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
            [
                beignet.polynomial._chebint.integrate_chebyshev_series(c, k=3)
                for c in c2d
            ]
        )
        res = beignet.polynomial._chebint.integrate_chebyshev_series(c2d, k=3, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)


class TestDerivative:
    def test_chebder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._chebder.differentiate_chebyshev_series,
            [0],
            0.5,
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._chebder.differentiate_chebyshev_series,
            [0],
            -1,
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.polynomial._chebder.differentiate_chebyshev_series(tgt, m=0)
            numpy.testing.assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial._chebder.differentiate_chebyshev_series(
                    beignet.polynomial._chebint.integrate_chebyshev_series(tgt, m=j),
                    m=j,
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial._chebder.differentiate_chebyshev_series(
                    beignet.polynomial._chebint.integrate_chebyshev_series(
                        tgt, m=j, scl=2
                    ),
                    m=j,
                    scl=0.5,
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_chebder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack(
            [
                beignet.polynomial._chebder.differentiate_chebyshev_series(c)
                for c in c2d.T
            ]
        ).T
        res = beignet.polynomial._chebder.differentiate_chebyshev_series(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack(
            [beignet.polynomial._chebder.differentiate_chebyshev_series(c) for c in c2d]
        )
        res = beignet.polynomial._chebder.differentiate_chebyshev_series(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_chebvander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.polynomial._chebvander.chebyshev_series_vandermonde(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i],
                beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef),
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.polynomial._chebvander.chebyshev_series_vandermonde(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i],
                beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef),
            )

    def test_chebvander2d(self):
        # also tests chebval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.polynomial._chebvander2d.chebvander2d(x1, x2, [1, 2])
        tgt = beignet.polynomial._chebval2d.evaluate_2d_chebyshev_series(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._chebvander2d.chebvander2d([x1], [x2], [1, 2])
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_chebvander3d(self):
        # also tests chebval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.polynomial._chebvander3d.chebvander3d(x1, x2, x3, [1, 2, 3])
        tgt = beignet.polynomial._chebval3d.evaluate_3d_chebyshev_series(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._chebvander3d.chebvander3d([x1], [x2], [x3], [1, 2, 3])
        numpy.testing.assert_(van.shape == (1, 5, 24))


class TestFitting:
    def test_chebfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebfit.fit_chebyshev_series, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebfit.fit_chebyshev_series, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebfit.fit_chebyshev_series, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebfit.fit_chebyshev_series, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebfit.fit_chebyshev_series, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebfit.fit_chebyshev_series, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._chebfit.fit_chebyshev_series,
            [1],
            [1],
            0,
            w=[[1]],
        )
        numpy.testing.assert_raises(
            TypeError,
            beignet.polynomial._chebfit.fit_chebyshev_series,
            [1],
            [1],
            0,
            w=[1, 1],
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._chebfit.fit_chebyshev_series,
            [1],
            [1],
            [
                -1,
            ],
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._chebfit.fit_chebyshev_series,
            [1],
            [1],
            [2, -1, 6],
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebfit.fit_chebyshev_series, [1], [1], []
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef3), y
        )
        coef3 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, [0, 1, 2, 3])
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef3), y
        )
        #
        coef4 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef4), y
        )
        coef4 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, [0, 1, 2, 3, 4])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef4), y
        )
        # check things still work if deg is not in strict increasing
        coef4 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, [2, 3, 4, 1, 0])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef4), y
        )
        #
        coef2d = beignet.polynomial._chebfit.fit_chebyshev_series(
            x, numpy.array([y, y]).T, 3
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.polynomial._chebfit.fit_chebyshev_series(
            x, numpy.array([y, y]).T, [0, 1, 2, 3]
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.polynomial._chebfit.fit_chebyshev_series(x, yw, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.polynomial._chebfit.fit_chebyshev_series(
            x, yw, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.polynomial._chebfit.fit_chebyshev_series(
            x, numpy.array([yw, yw]).T, 3, w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.polynomial._chebfit.fit_chebyshev_series(
            x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebfit.fit_chebyshev_series(x, x, 1), [0, 1]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebfit.fit_chebyshev_series(x, x, [0, 1]), [0, 1]
        )
        # test fitting only even polynomials
        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef1), y
        )
        coef2 = beignet.polynomial._chebfit.fit_chebyshev_series(x, y, [0, 2, 4])
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebval.evaluate_chebyshev_series(x, coef2), y
        )
        numpy.testing.assert_almost_equal(coef1, coef2)


class TestInterpolate:
    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebinterpolate.chebinterpolate, self.f, -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._chebinterpolate.chebinterpolate, self.f, 10.0
        )

    def test_dimensions(self):
        for deg in range(1, 5):
            numpy.testing.assert_(
                beignet.polynomial._chebinterpolate.chebinterpolate(self.f, deg).shape
                == (deg + 1,)
            )

    def test_approximation(self):
        def powx(x, p):
            return x**p

        x = numpy.linspace(-1, 1, 10)
        for deg in range(0, 10):
            for p in range(0, deg + 1):
                c = beignet.polynomial._chebinterpolate.chebinterpolate(powx, deg, (p,))
                numpy.testing.assert_almost_equal(
                    beignet.polynomial._chebval.evaluate_chebyshev_series(x, c),
                    powx(x, p),
                    decimal=12,
                )


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebcompanion.chebcompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebcompanion.chebcompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.polynomial._chebcompanion.chebcompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.polynomial._chebcompanion.chebcompanion([1, 2])[0, 0] == -0.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.polynomial._chebgauss.chebyshev_gauss_quadrature(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = beignet.polynomial._chebvander.chebyshev_series_vandermonde(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_almost_equal(vv, numpy.eye(100))

        # check that the integral of 1 is correct
        tgt = numpy.pi
        numpy.testing.assert_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_chebfromroots(self):
        res = beignet.polynomial._chebfromroots.chebyshev_series_from_roots([])
        numpy.testing.assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            tgt = [0] * i + [1]
            res = beignet.polynomial._chebfromroots.chebyshev_series_from_roots(
                roots
            ) * 2 ** (i - 1)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_chebroots(self):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebroots.chebyshev_series_roots([1]), []
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebroots.chebyshev_series_roots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.polynomial._chebroots.chebyshev_series_roots(
                beignet.polynomial._chebfromroots.chebyshev_series_from_roots(tgt)
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_chebtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebtrim.trim_chebyshev_series, coef, -1
        )

        # Test results
        numpy.testing.assert_equal(
            beignet.polynomial._chebtrim.trim_chebyshev_series(coef), coef[:-1]
        )
        numpy.testing.assert_equal(
            beignet.polynomial._chebtrim.trim_chebyshev_series(coef, 1), coef[:-3]
        )
        numpy.testing.assert_equal(
            beignet.polynomial._chebtrim.trim_chebyshev_series(coef, 2), [0]
        )

    def test_chebline(self):
        numpy.testing.assert_equal(beignet.polynomial._chebline.chebline(3, 4), [3, 4])

    def test_cheb2poly(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._cheb2poly.chebyshev_series_to_polynomial_series(
                    [0] * i + [1]
                ),
                Tlist[i],
            )

    def test_poly2cheb(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._poly2cheb.power_series_to_chebyshev_series(
                    Tlist[i]
                ),
                [0] * i + [1],
            )

    def test_weight(self):
        x = numpy.linspace(-1, 1, 11)[1:-1]
        tgt = 1.0 / (numpy.sqrt(1 + x) * numpy.sqrt(1 - x))
        res = beignet.polynomial._chebweight.chebweight(x)
        numpy.testing.assert_almost_equal(res, tgt)

    def test_chebpts1(self):
        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebpts1.chebpts1, 1.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebpts1.chebpts1, 0
        )

        # test points
        tgt = [0]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts1.chebpts1(1), tgt)
        tgt = [-0.70710678118654746, 0.70710678118654746]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts1.chebpts1(2), tgt)
        tgt = [-0.86602540378443871, 0, 0.86602540378443871]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts1.chebpts1(3), tgt)
        tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts1.chebpts1(4), tgt)

    def test_chebpts2(self):
        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebpts2.chebpts2, 1.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._chebpts2.chebpts2, 1
        )

        # test points
        tgt = [-1, 1]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts2.chebpts2(2), tgt)
        tgt = [-1, 0, 1]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts2.chebpts2(3), tgt)
        tgt = [-1, -0.5, 0.5, 1]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts2.chebpts2(4), tgt)
        tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
        numpy.testing.assert_almost_equal(beignet.polynomial._chebpts2.chebpts2(5), tgt)
