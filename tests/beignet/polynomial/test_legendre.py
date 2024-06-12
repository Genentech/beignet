import functools

import beignet.polynomial
import beignet.polynomial.legendre
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
    return beignet.polynomial.legendre.legtrim(x, tol=1e-6)


class TestConstants:
    def test_legdomain(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legdomain, [-1, 1])

    def test_legzero(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legzero, [0])

    def test_legone(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legone, [1])

    def test_legx(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legx, [0, 1])


class TestArithmetic:
    x = numpy.linspace(-1, 1, 100)

    def test_legadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.polynomial.legendre.legadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.polynomial.legendre.legsub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legmulx([0]), [0])
        numpy.testing.assert_equal(beignet.polynomial.legendre.legmulx([1]), [0, 1])
        for i in range(1, 5):
            tmp = 2 * i + 1
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
            numpy.testing.assert_equal(beignet.polynomial.legendre.legmulx(ser), tgt)

    def test_legmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.polynomial.legendre.legval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = beignet.polynomial.legendre.legval(self.x, pol2)
                pol3 = beignet.polynomial.legendre.legmul(pol1, pol2)
                val3 = beignet.polynomial.legendre.legval(self.x, pol3)
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_legdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.polynomial.legendre.legadd(ci, cj)
                quo, rem = beignet.polynomial.legendre.legdiv(tgt, ci)
                res = beignet.polynomial.legendre.legadd(
                    beignet.polynomial.legendre.legmul(quo, ci), rem
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.polynomial.legendre.legmul, [c] * j, numpy.array([1])
                )
                res = beignet.polynomial.legendre.legpow(c, j)
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_legval(self):
        # check empty input
        numpy.testing.assert_equal(beignet.polynomial.legendre.legval([], [1]).size, 0)

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [beignet.polynomial.polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.polynomial.legendre.legval(x, [0] * i + [1])
            numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial.legendre.legval(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial.legendre.legval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial.legendre.legval(x, [1, 0, 0]).shape, dims
            )

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legval2d, x1, x2[:2], self.c2d
        )

        # test values
        tgt = y1 * y2
        res = beignet.polynomial.legendre.legval2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.legendre.legval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legval3d, x1, x2, x3[:2], self.c3d
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.polynomial.legendre.legval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.legendre.legval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.polynomial.legendre.leggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.legendre.leggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.polynomial.legendre.leggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.legendre.leggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_legint(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legint, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legint, [0], -1
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legint, [0], 1, [0, 0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legint, [0], lbnd=[0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legint, [0], scl=[0]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legint, [0], axis=0.5
        )

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.polynomial.legendre.legint([0], m=i, k=k)
            numpy.testing.assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            legpol = beignet.polynomial.legendre.poly2leg(pol)
            legint = beignet.polynomial.legendre.legint(legpol, m=1, k=[i])
            res = beignet.polynomial.legendre.leg2poly(legint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            legpol = beignet.polynomial.legendre.poly2leg(pol)
            legint = beignet.polynomial.legendre.legint(legpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legendre.legval(-1, legint), i
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            legpol = beignet.polynomial.legendre.poly2leg(pol)
            legint = beignet.polynomial.legendre.legint(legpol, m=1, k=[i], scl=2)
            res = beignet.polynomial.legendre.leg2poly(legint)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.polynomial.legendre.legint(tgt, m=1)
                res = beignet.polynomial.legendre.legint(pol, m=j)
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial.legendre.legint(tgt, m=1, k=[k])
                res = beignet.polynomial.legendre.legint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial.legendre.legint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.polynomial.legendre.legint(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial.legendre.legint(tgt, m=1, k=[k], scl=2)
                res = beignet.polynomial.legendre.legint(
                    pol, m=j, k=list(range(j)), scl=2
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_legint_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial.legendre.legint(c) for c in c2d.T]).T
        res = beignet.polynomial.legendre.legint(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial.legendre.legint(c) for c in c2d])
        res = beignet.polynomial.legendre.legint(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial.legendre.legint(c, k=3) for c in c2d])
        res = beignet.polynomial.legendre.legint(c2d, k=3, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        numpy.testing.assert_equal(
            beignet.polynomial.legendre.legint((1, 2, 3), 0), (1, 2, 3)
        )


class TestDerivative:
    def test_legder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legder, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legder, [0], -1
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.legendre.legder(tgt, m=0)
            numpy.testing.assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial.legendre.legder(
                    beignet.polynomial.legendre.legint(tgt, m=j), m=j
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial.legendre.legder(
                    beignet.polynomial.legendre.legint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial.legendre.legder(c) for c in c2d.T]).T
        res = beignet.polynomial.legendre.legder(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial.legendre.legder(c) for c in c2d])
        res = beignet.polynomial.legendre.legder(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        numpy.testing.assert_equal(beignet.polynomial.legendre.legder(c, 4), [0])


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_legvander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.polynomial.legendre.legvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial.legendre.legval(x, coef)
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.polynomial.legendre.legvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial.legendre.legval(x, coef)
            )

    def test_legvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.polynomial.legendre.legvander2d(x1, x2, [1, 2])
        tgt = beignet.polynomial.legendre.legval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial.legendre.legvander2d([x1], [x2], [1, 2])
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_legvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.polynomial.legendre.legvander3d(x1, x2, x3, [1, 2, 3])
        tgt = beignet.polynomial.legendre.legval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial.legendre.legvander3d([x1], [x2], [x3], [1, 2, 3])
        numpy.testing.assert_(van.shape == (1, 5, 24))

    def test_legvander_negdeg(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legvander, (1, 2, 3), -1
        )


class TestFitting:
    def test_legfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legfit, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial.legendre.legfit,
            [1],
            [1],
            [
                -1,
            ],
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legfit, [1], [1], [2, -1, 6]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.legendre.legfit, [1], [1], []
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.polynomial.legendre.legfit(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef3), y
        )
        coef3 = beignet.polynomial.legendre.legfit(x, y, [0, 1, 2, 3])
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef3), y
        )
        #
        coef4 = beignet.polynomial.legendre.legfit(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef4), y
        )
        coef4 = beignet.polynomial.legendre.legfit(x, y, [0, 1, 2, 3, 4])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef4), y
        )
        # check things still work if deg is not in strict increasing
        coef4 = beignet.polynomial.legendre.legfit(x, y, [2, 3, 4, 1, 0])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef4), y
        )
        #
        coef2d = beignet.polynomial.legendre.legfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.polynomial.legendre.legfit(
            x, numpy.array([y, y]).T, [0, 1, 2, 3]
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.polynomial.legendre.legfit(x, yw, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.polynomial.legendre.legfit(x, yw, [0, 1, 2, 3], w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.polynomial.legendre.legfit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.polynomial.legendre.legfit(
            x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legfit(x, x, 1), [0, 1]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legfit(x, x, [0, 1]), [0, 1]
        )
        # test fitting only even Legendre polynomials
        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.polynomial.legendre.legfit(x, y, 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef1), y
        )
        coef2 = beignet.polynomial.legendre.legfit(x, y, [0, 2, 4])
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legval(x, coef2), y
        )
        numpy.testing.assert_almost_equal(coef1, coef2)


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legcompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legcompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.polynomial.legendre.legcompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.polynomial.legendre.legcompanion([1, 2])[0, 0] == -0.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.polynomial.legendre.leggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = beignet.polynomial.legendre.legvander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_almost_equal(vv, numpy.eye(100))

        # check that the integral of 1 is correct
        tgt = 2.0
        numpy.testing.assert_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_legfromroots(self):
        res = beignet.polynomial.legendre.legfromroots([])
        numpy.testing.assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            pol = beignet.polynomial.legendre.legfromroots(roots)
            res = beignet.polynomial.legendre.legval(roots, pol)
            tgt = 0
            numpy.testing.assert_(len(pol) == i + 1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legendre.leg2poly(pol)[-1], 1
            )
            numpy.testing.assert_almost_equal(res, tgt)

    def test_legroots(self):
        numpy.testing.assert_almost_equal(beignet.polynomial.legendre.legroots([1]), [])
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legendre.legroots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.polynomial.legendre.legroots(
                beignet.polynomial.legendre.legfromroots(tgt)
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_legtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.legendre.legtrim, coef, -1
        )

        # Test results
        numpy.testing.assert_equal(beignet.polynomial.legendre.legtrim(coef), coef[:-1])
        numpy.testing.assert_equal(
            beignet.polynomial.legendre.legtrim(coef, 1), coef[:-3]
        )
        numpy.testing.assert_equal(beignet.polynomial.legendre.legtrim(coef, 2), [0])

    def test_legline(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legline(3, 4), [3, 4])

    def test_legline_zeroscl(self):
        numpy.testing.assert_equal(beignet.polynomial.legendre.legline(3, 0), [3])

    def test_leg2poly(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legendre.leg2poly([0] * i + [1]), Llist[i]
            )

    def test_poly2leg(self):
        for i in range(10):
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legendre.poly2leg(Llist[i]), [0] * i + [1]
            )

    def test_weight(self):
        x = numpy.linspace(-1, 1, 11)
        tgt = 1.0
        res = beignet.polynomial.legendre.legweight(x)
        numpy.testing.assert_almost_equal(res, tgt)
