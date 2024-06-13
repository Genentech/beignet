import functools

import beignet.polynomial
import beignet.polynomial._lag2poly
import beignet.polynomial._lagadd
import beignet.polynomial._lagcompanion
import beignet.polynomial._lagder
import beignet.polynomial._lagdiv
import beignet.polynomial._lagdomain
import beignet.polynomial._lagfit
import beignet.polynomial._lagfromroots
import beignet.polynomial._laggauss
import beignet.polynomial._laggrid2d
import beignet.polynomial._laggrid3d
import beignet.polynomial._lagint
import beignet.polynomial._lagline
import beignet.polynomial._lagmul
import beignet.polynomial._lagmulx
import beignet.polynomial._lagone
import beignet.polynomial._lagpow
import beignet.polynomial._lagroots
import beignet.polynomial._lagsub
import beignet.polynomial._lagtrim
import beignet.polynomial._lagval
import beignet.polynomial._lagval2d
import beignet.polynomial._lagval3d
import beignet.polynomial._lagvander
import beignet.polynomial._lagvander2d
import beignet.polynomial._lagvander3d
import beignet.polynomial._lagweight
import beignet.polynomial._lagx
import beignet.polynomial._lagzero
import beignet.polynomial._poly2lag
import beignet.polynomial._polyval
import numpy
import numpy.testing

L0 = numpy.array([1]) / 1
L1 = numpy.array([1, -1]) / 1
L2 = numpy.array([2, -4, 1]) / 2
L3 = numpy.array([6, -18, 9, -1]) / 6
L4 = numpy.array([24, -96, 72, -16, 1]) / 24
L5 = numpy.array([120, -600, 600, -200, 25, -1]) / 120
L6 = numpy.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720

Llist = [L0, L1, L2, L3, L4, L5, L6]


def trim(x):
    return beignet.polynomial._lagtrim.lagtrim(x, tol=1e-6)


class TestConstants:
    def test_lagdomain(self):
        numpy.testing.assert_equal(beignet.polynomial._lagdomain.lagdomain, [0, 1])

    def test_lagzero(self):
        numpy.testing.assert_equal(beignet.polynomial._lagzero.lagzero, [0])

    def test_lagone(self):
        numpy.testing.assert_equal(beignet.polynomial._lagone.lagone, [1])

    def test_lagx(self):
        numpy.testing.assert_equal(beignet.polynomial._lagx.lagx, [1, -1])


class TestArithmetic:
    x = numpy.linspace(-3, 3, 100)

    def test_lagadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.polynomial._lagadd.lagadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.polynomial._lagsub.lagsub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagmulx(self):
        numpy.testing.assert_equal(beignet.polynomial._lagmulx.lagmulx([0]), [0])
        numpy.testing.assert_equal(beignet.polynomial._lagmulx.lagmulx([1]), [1, -1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagmulx.lagmulx(ser), tgt
            )

    def test_lagmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.polynomial._lagval.lagval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = beignet.polynomial._lagval.lagval(self.x, pol2)
                pol3 = beignet.polynomial._lagmul.lagmul(pol1, pol2)
                val3 = beignet.polynomial._lagval.lagval(self.x, pol3)
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_lagdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.polynomial._lagadd.lagadd(ci, cj)
                quo, rem = beignet.polynomial._lagdiv.lagdiv(tgt, ci)
                res = beignet.polynomial._lagadd.lagadd(
                    beignet.polynomial._lagmul.lagmul(quo, ci), rem
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.polynomial._lagmul.lagmul, [c] * j, numpy.array([1])
                )
                res = beignet.polynomial._lagpow.lagpow(c, j)
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.polyval(x, [1.0, 2.0, 3.0])

    def test_lagval(self):
        # check empty input
        numpy.testing.assert_equal(beignet.polynomial._lagval.lagval([], [1]).size, 0)

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [beignet.polynomial._polyval.polyval(x, c) for c in Llist]
        for i in range(7):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.polynomial._lagval.lagval(x, [0] * i + [1])
            numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial._lagval.lagval(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial._lagval.lagval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial._lagval.lagval(x, [1, 0, 0]).shape, dims
            )

    def test_lagval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagval2d.lagval2d, x1, x2[:2], self.c2d
        )

        # test values
        tgt = y1 * y2
        res = beignet.polynomial._lagval2d.lagval2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._lagval2d.lagval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_lagval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagval3d.lagval3d, x1, x2, x3[:2], self.c3d
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.polynomial._lagval3d.lagval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._lagval3d.lagval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_laggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.polynomial._laggrid2d.laggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._laggrid2d.laggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_laggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.polynomial._laggrid3d.laggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial._laggrid3d.laggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_lagint(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagint.lagint, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagint.lagint, [0], -1
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagint.lagint, [0], 1, [0, 0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagint.lagint, [0], lbnd=[0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagint.lagint, [0], scl=[0]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagint.lagint, [0], axis=0.5
        )

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.polynomial._lagint.lagint([0], m=i, k=k)
            numpy.testing.assert_almost_equal(res, [1, -1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            lagpol = beignet.polynomial._poly2lag.poly2lag(pol)
            lagint = beignet.polynomial._lagint.lagint(lagpol, m=1, k=[i])
            res = beignet.polynomial._lag2poly.lag2poly(lagint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            lagpol = beignet.polynomial._poly2lag.poly2lag(pol)
            lagint = beignet.polynomial._lagint.lagint(lagpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagval.lagval(-1, lagint), i
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            lagpol = beignet.polynomial._poly2lag.poly2lag(pol)
            lagint = beignet.polynomial._lagint.lagint(lagpol, m=1, k=[i], scl=2)
            res = beignet.polynomial._lag2poly.lag2poly(lagint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.polynomial._lagint.lagint(tgt, m=1)
                res = beignet.polynomial._lagint.lagint(pol, m=j)
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial._lagint.lagint(tgt, m=1, k=[k])
                res = beignet.polynomial._lagint.lagint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial._lagint.lagint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.polynomial._lagint.lagint(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial._lagint.lagint(tgt, m=1, k=[k], scl=2)
                res = beignet.polynomial._lagint.lagint(
                    pol, m=j, k=list(range(j)), scl=2
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_lagint_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial._lagint.lagint(c) for c in c2d.T]).T
        res = beignet.polynomial._lagint.lagint(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial._lagint.lagint(c) for c in c2d])
        res = beignet.polynomial._lagint.lagint(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial._lagint.lagint(c, k=3) for c in c2d])
        res = beignet.polynomial._lagint.lagint(c2d, k=3, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)


class TestDerivative:
    def test_lagder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagder.lagder, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagder.lagder, [0], -1
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.polynomial._lagder.lagder(tgt, m=0)
            numpy.testing.assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial._lagder.lagder(
                    beignet.polynomial._lagint.lagint(tgt, m=j), m=j
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial._lagder.lagder(
                    beignet.polynomial._lagint.lagint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_lagder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial._lagder.lagder(c) for c in c2d.T]).T
        res = beignet.polynomial._lagder.lagder(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial._lagder.lagder(c) for c in c2d])
        res = beignet.polynomial._lagder.lagder(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_lagvander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.polynomial._lagvander.lagvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial._lagval.lagval(x, coef)
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.polynomial._lagvander.lagvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial._lagval.lagval(x, coef)
            )

    def test_lagvander2d(self):
        # also tests lagval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.polynomial._lagvander2d.lagvander2d(x1, x2, [1, 2])
        tgt = beignet.polynomial._lagval2d.lagval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._lagvander2d.lagvander2d([x1], [x2], [1, 2])
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_lagvander3d(self):
        # also tests lagval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.polynomial._lagvander3d.lagvander3d(x1, x2, x3, [1, 2, 3])
        tgt = beignet.polynomial._lagval3d.lagval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial._lagvander3d.lagvander3d([x1], [x2], [x3], [1, 2, 3])
        numpy.testing.assert_(van.shape == (1, 5, 24))


class TestFitting:
    def test_lagfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagfit.lagfit, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial._lagfit.lagfit,
            [1],
            [1],
            [
                -1,
            ],
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagfit.lagfit, [1], [1], [2, -1, 6]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial._lagfit.lagfit, [1], [1], []
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.polynomial._lagfit.lagfit(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.lagval(x, coef3), y
        )
        coef3 = beignet.polynomial._lagfit.lagfit(x, y, [0, 1, 2, 3])
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.lagval(x, coef3), y
        )
        #
        coef4 = beignet.polynomial._lagfit.lagfit(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.lagval(x, coef4), y
        )
        coef4 = beignet.polynomial._lagfit.lagfit(x, y, [0, 1, 2, 3, 4])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.lagval(x, coef4), y
        )
        #
        coef2d = beignet.polynomial._lagfit.lagfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.polynomial._lagfit.lagfit(
            x, numpy.array([y, y]).T, [0, 1, 2, 3]
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.polynomial._lagfit.lagfit(x, yw, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.polynomial._lagfit.lagfit(x, yw, [0, 1, 2, 3], w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.polynomial._lagfit.lagfit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.polynomial._lagfit.lagfit(
            x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagfit.lagfit(x, x, 1), [1, -1]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagfit.lagfit(x, x, [0, 1]), [1, -1]
        )


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagcompanion.lagcompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagcompanion.lagcompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.polynomial._lagcompanion.lagcompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.polynomial._lagcompanion.lagcompanion([1, 2])[0, 0] == 1.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.polynomial._laggauss.laggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = beignet.polynomial._lagvander.lagvander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_almost_equal(vv, numpy.eye(100))

        # check that the integral of 1 is correct
        tgt = 1.0
        numpy.testing.assert_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_lagfromroots(self):
        res = beignet.polynomial._lagfromroots.lagfromroots([])
        numpy.testing.assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            pol = beignet.polynomial._lagfromroots.lagfromroots(roots)
            res = beignet.polynomial._lagval.lagval(roots, pol)
            tgt = 0
            numpy.testing.assert_(len(pol) == i + 1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lag2poly.lag2poly(pol)[-1], 1
            )
            numpy.testing.assert_almost_equal(res, tgt)

    def test_lagroots(self):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagroots.lagroots([1]), []
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagroots.lagroots([0, 1]), [1]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(0, 3, i)
            res = beignet.polynomial._lagroots.lagroots(
                beignet.polynomial._lagfromroots.lagfromroots(tgt)
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_lagtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial._lagtrim.lagtrim, coef, -1
        )

        # Test results
        numpy.testing.assert_equal(beignet.polynomial._lagtrim.lagtrim(coef), coef[:-1])
        numpy.testing.assert_equal(
            beignet.polynomial._lagtrim.lagtrim(coef, 1), coef[:-3]
        )
        numpy.testing.assert_equal(beignet.polynomial._lagtrim.lagtrim(coef, 2), [0])

    def test_lagline(self):
        numpy.testing.assert_equal(beignet.polynomial._lagline.lagline(3, 4), [7, -4])

    def test_lag2poly(self):
        for i in range(7):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lag2poly.lag2poly([0] * i + [1]), Llist[i]
            )

    def test_poly2lag(self):
        for i in range(7):
            numpy.testing.assert_almost_equal(
                beignet.polynomial._poly2lag.poly2lag(Llist[i]), [0] * i + [1]
            )

    def test_weight(self):
        x = numpy.linspace(0, 10, 11)
        tgt = numpy.exp(-x)
        res = beignet.polynomial._lagweight.lagweight(x)
        numpy.testing.assert_almost_equal(res, tgt)
