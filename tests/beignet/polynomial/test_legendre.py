import functools

import beignet.polynomial
import beignet.polynomial._add_legendre_series
import beignet.polynomial._leg2poly
import beignet.polynomial._legcompanion
import beignet.polynomial._legder
import beignet.polynomial._legdiv
import beignet.polynomial._legdomain
import beignet.polynomial._legfit
import beignet.polynomial._legfromroots
import beignet.polynomial._leggauss
import beignet.polynomial._leggrid2d
import beignet.polynomial._leggrid3d
import beignet.polynomial._legint
import beignet.polynomial._legline
import beignet.polynomial._legmul
import beignet.polynomial._legmulx
import beignet.polynomial._legone
import beignet.polynomial._legpow
import beignet.polynomial._legroots
import beignet.polynomial._legsub
import beignet.polynomial._legtrim
import beignet.polynomial._legval
import beignet.polynomial._legval2d
import beignet.polynomial._legval3d
import beignet.polynomial._legvander
import beignet.polynomial._legvander2d
import beignet.polynomial._legvander3d
import beignet.polynomial._legweight
import beignet.polynomial._legx
import beignet.polynomial._legzero
import beignet.polynomial._poly2leg
import beignet.polynomial._polyval
import numpy
import numpy.testing

L0 = numpy.array([1])
L1 = numpy.array([0, 1])
L2 = numpy.array([-1, 0, 3]) / 2
L3 = numpy.array([0, -3, 0, 5]) / 2
L4 = numpy.array([3, 0, -30, 0, 35]) / 8
L5 = numpy.array([0, 15, 0, -70, 0, 63]) / 8
L6 = numpy.array([-5, 0, 105, 0, -315, 0, 231]) / 16
L7 = numpy.array([0, -35, 0, 315, 0, -693, 0, 429]) / 16
L8 = numpy.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128
L9 = numpy.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128

Llist = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]


def trim(x):
    return beignet.polynomial._legtrim.legtrim(x, tol=1e-6)


class TestConstants:
    def test_legdomain(self):
        numpy.testing.assert_equal(beignet.polynomial._legdomain.legdomain, [-1, 1])

    def test_legzero(self):
        numpy.testing.assert_equal(beignet.polynomial._legzero.legzero, [0])

    def test_legone(self):
        numpy.testing.assert_equal(beignet.polynomial._legone.legone, [1])

    def test_legx(self):
        numpy.testing.assert_equal(beignet.polynomial._legx.legx, [0, 1])


class TestArithmetic:
    x = numpy.linspace(-1, 1, 100)

    def test_legadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.polynomial._legadd.add_legendre_series(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.polynomial._legsub.legsub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        numpy.testing.assert_equal(beignet.polynomial._legmulx.legmulx([0]), [0])
        numpy.testing.assert_equal(beignet.polynomial._legmulx.legmulx([1]), [0, 1])
        for i in range(1, 5):
            tmp = 2 * i + 1
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
            numpy.testing.assert_equal(beignet.polynomial._legmulx.legmulx(ser), tgt)

    def test_legmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.polynomial._legval.legval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = beignet.polynomial._legval.legval(self.x, pol2)
                pol3 = beignet.polynomial._legmul.legmul(pol1, pol2)
                val3 = beignet.polynomial._legval.legval(self.x, pol3)
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_legdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.polynomial._legadd.add_legendre_series(ci, cj)
                quo, rem = beignet.polynomial._legdiv.legdiv(tgt, ci)
                res = beignet.polynomial._legadd.add_legendre_series(
                    beignet.polynomial._legmul.legmul(quo, ci), rem
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.polynomial._legmul.legmul, [c] * j, numpy.array([1])
                )
                res = beignet.polynomial._legpow.legpow(c, j)
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.polyval(x, [1.0, 2.0, 3.0])

    def test_legval(self):
        # check empty input
        numpy.testing.assert_equal(beignet.polynomial._legval.legval([], [1]).size, 0)

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [beignet.polynomial._polyval.polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.polynomial._legval.legval(x, [0] * i + [1])
            numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial._legval.legval(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial._legval.legval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial._legval.legval(x, [1, 0, 0]).shape, dims
            )

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legval2d.legval2d, x1, x2[:2], self.c2d
        )

        # test values
        tgt = y1 * y2
        res = beignet.polynomial._legval2d.legval2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._legval2d.legval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legval3d.legval3d, x1, x2, x3[:2], self.c3d
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.polynomial._legval3d.legval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._legval3d.legval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.polynomial._leggrid2d.leggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._leggrid2d.leggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.polynomial._leggrid3d.leggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._leggrid3d.leggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_legint(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legint.legint, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legint.legint, [0], -1
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legint.legint, [0], 1, [0, 0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legint.legint, [0], lbnd=[0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legint.legint, [0], scl=[0]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legint.legint, [0], axis=0.5
        )

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.polynomial._legint.legint([0], m=i, k=k)
            numpy.testing.assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            legpol = beignet.polynomial._poly2leg.poly2leg(pol)
            legint = beignet.polynomial._legint.legint(legpol, m=1, k=[i])
            res = beignet.polynomial._leg2poly.leg2poly(legint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            legpol = beignet.polynomial._poly2leg.poly2leg(pol)
            legint = beignet.polynomial._legint.legint(legpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._legval.legval(-1, legint), i
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            legpol = beignet.polynomial._poly2leg.poly2leg(pol)
            legint = beignet.polynomial._legint.legint(legpol, m=1, k=[i], scl=2)
            res = beignet.polynomial._leg2poly.leg2poly(legint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.polynomial._legint.legint(tgt, m=1)
                res = beignet.polynomial._legint.legint(pol, m=j)
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial._legint.legint(tgt, m=1, k=[k])
                res = beignet.polynomial._legint.legint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial._legint.legint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.polynomial._legint.legint(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial._legint.legint(tgt, m=1, k=[k], scl=2)
                res = beignet.polynomial._legint.legint(
                    pol, m=j, k=list(range(j)), scl=2
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_legint_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial._legint.legint(c) for c in c2d.T]).T
        res = beignet.polynomial._legint.legint(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial._legint.legint(c) for c in c2d])
        res = beignet.polynomial._legint.legint(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial._legint.legint(c, k=3) for c in c2d])
        res = beignet.polynomial._legint.legint(c2d, k=3, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        numpy.testing.assert_equal(
            beignet.polynomial._legint.legint((1, 2, 3), 0), (1, 2, 3)
        )


class TestDerivative:
    def test_legder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legder.legder, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legder.legder, [0], -1
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.polynomial._legder.legder(tgt, m=0)
            numpy.testing.assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial._legder.legder(
                    beignet.polynomial._legint.legint(tgt, m=j), m=j
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial._legder.legder(
                    beignet.polynomial._legint.legint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial._legder.legder(c) for c in c2d.T]).T
        res = beignet.polynomial._legder.legder(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial._legder.legder(c) for c in c2d])
        res = beignet.polynomial._legder.legder(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        numpy.testing.assert_equal(beignet.polynomial._legder.legder(c, 4), [0])


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_legvander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.polynomial._legvander.legvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial._legval.legval(x, coef)
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.polynomial._legvander.legvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial._legval.legval(x, coef)
            )

    def test_legvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.polynomial._legvander2d.legvander2d(x1, x2, [1, 2])
        tgt = beignet.polynomial._legval2d.legval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._legvander2d.legvander2d([x1], [x2], [1, 2])
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_legvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.polynomial._legvander3d.legvander3d(x1, x2, x3, [1, 2, 3])
        tgt = beignet.polynomial._legval3d.legval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._legvander3d.legvander3d([x1], [x2], [x3], [1, 2, 3])
        numpy.testing.assert_(van.shape == (1, 5, 24))

    def test_legvander_negdeg(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legvander.legvander, (1, 2, 3), -1
        )


class TestFitting:
    def test_legfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legfit.legfit, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._legfit.legfit,
            [1],
            [1],
            [
                -1,
            ],
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legfit.legfit, [1], [1], [2, -1, 6]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._legfit.legfit, [1], [1], []
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.polynomial._legfit.legfit(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef3), y
        )
        coef3 = beignet.polynomial._legfit.legfit(x, y, [0, 1, 2, 3])
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef3), y
        )
        #
        coef4 = beignet.polynomial._legfit.legfit(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef4), y
        )
        coef4 = beignet.polynomial._legfit.legfit(x, y, [0, 1, 2, 3, 4])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef4), y
        )
        # check things still work if deg is not in strict increasing
        coef4 = beignet.polynomial._legfit.legfit(x, y, [2, 3, 4, 1, 0])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef4), y
        )
        #
        coef2d = beignet.polynomial._legfit.legfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.polynomial._legfit.legfit(
            x, numpy.array([y, y]).T, [0, 1, 2, 3]
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.polynomial._legfit.legfit(x, yw, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.polynomial._legfit.legfit(x, yw, [0, 1, 2, 3], w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.polynomial._legfit.legfit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.polynomial._legfit.legfit(
            x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legfit.legfit(x, x, 1), [0, 1]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legfit.legfit(x, x, [0, 1]), [0, 1]
        )
        # test fitting only even Legendre polynomials
        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.polynomial._legfit.legfit(x, y, 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef1), y
        )
        coef2 = beignet.polynomial._legfit.legfit(x, y, [0, 2, 4])
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legval.legval(x, coef2), y
        )
        numpy.testing.assert_almost_equal(coef1, coef2)


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legcompanion.legcompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legcompanion.legcompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.polynomial._legcompanion.legcompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.polynomial._legcompanion.legcompanion([1, 2])[0, 0] == -0.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.polynomial._leggauss.leggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = beignet.polynomial._legvander.legvander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_almost_equal(vv, numpy.eye(100))

        # check that the integral of 1 is correct
        tgt = 2.0
        numpy.testing.assert_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_legfromroots(self):
        res = beignet.polynomial._legfromroots.legfromroots([])
        numpy.testing.assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            pol = beignet.polynomial._legfromroots.legfromroots(roots)
            res = beignet.polynomial._legval.legval(roots, pol)
            tgt = 0
            numpy.testing.assert_(len(pol) == i + 1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._leg2poly.leg2poly(pol)[-1], 1
            )
            numpy.testing.assert_almost_equal(res, tgt)

    def test_legroots(self):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legroots.legroots([1]), []
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legroots.legroots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.polynomial._legroots.legroots(
                beignet.polynomial._legfromroots.legfromroots(tgt)
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_legtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._legtrim.legtrim, coef, -1
        )

        # Test results
        numpy.testing.assert_equal(beignet.polynomial._legtrim.legtrim(coef), coef[:-1])
        numpy.testing.assert_equal(
            beignet.polynomial._legtrim.legtrim(coef, 1), coef[:-3]
        )
        numpy.testing.assert_equal(beignet.polynomial._legtrim.legtrim(coef, 2), [0])

    def test_legline(self):
        numpy.testing.assert_equal(beignet.polynomial._legline.legline(3, 4), [3, 4])

    def test_legline_zeroscl(self):
        numpy.testing.assert_equal(beignet.polynomial._legline.legline(3, 0), [3])

    def test_leg2poly(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._leg2poly.leg2poly([0] * i + [1]), Llist[i]
            )

    def test_poly2leg(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._poly2leg.poly2leg(Llist[i]), [0] * i + [1]
            )

    def test_weight(self):
        x = numpy.linspace(-1, 1, 11)
        tgt = 1.0
        res = beignet.polynomial._legweight.legweight(x)
        numpy.testing.assert_almost_equal(res, tgt)
