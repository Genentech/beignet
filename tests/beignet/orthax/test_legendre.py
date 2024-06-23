import functools

import beignet.orthax.legendre
import numpy
import numpy.polynomial.polynomial
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
    return beignet.orthax.legendre.legtrim(x, tol=1e-6)


class TestConstants:
    def test_legdomain(self):
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legdomain, [-1, 1])

    def test_legzero(self):
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legzero, [0])

    def test_legone(self):
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legone, [1])

    def test_legx(self):
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legx, [0, 1])


class TestArithmetic:
    x = numpy.linspace(-1, 1, 100)

    def test_legadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.orthax.legendre.legadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.legendre.legsub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        numpy.testing.assert_array_equal(
            trim(beignet.orthax.legendre.legmulx([0])), [0]
        )
        numpy.testing.assert_array_equal(
            trim(beignet.orthax.legendre.legmulx([1])), [0, 1]
        )
        for i in range(1, 5):
            tmp = 2 * i + 1
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
            numpy.testing.assert_array_equal(
                trim(beignet.orthax.legendre.legmulx(ser)), tgt
            )

    def test_legmul(self):
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.orthax.legendre.legval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = beignet.orthax.legendre.legval(self.x, pol2)
                pol3 = beignet.orthax.legendre.legmul(pol1, pol2)
                val3 = beignet.orthax.legendre.legval(self.x, pol3)
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_legdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.orthax.legendre.legadd(ci, cj)
                quo, rem = beignet.orthax.legendre.legdiv(tgt, ci)
                res = beignet.orthax.legendre.legadd(
                    beignet.orthax.legendre.legmul(quo, ci), rem
                )
                numpy.testing.assert_array_almost_equal(
                    trim(res), trim(tgt), err_msg=msg
                )

    def test_legpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.orthax.legendre.legmul, [c] * j, numpy.array([1])
                )
                res = beignet.orthax.legendre.legpow(c, j)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_legval(self):
        numpy.testing.assert_array_equal(
            beignet.orthax.legendre.legval([], [1]).size, 0
        )

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.legendre.legval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.legendre.legval(x, [1]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.legendre.legval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.legendre.legval(x, [1, 0, 0]).shape, dims
            )

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.legendre.legval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legendre.legval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.legendre.legval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legendre.legval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.legendre.leggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legendre.leggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.legendre.leggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legendre.leggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_legint(self):  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.legendre.legint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legendre.legint, [0], -1)
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legint, [0], 1, [0, 0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legint, [0], lbnd=[0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legint, [0], scl=[0]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legint, [0], axis=0.5
        )

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.legendre.legint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(trim(res), [0, 1])

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            legpol = beignet.orthax.legendre.poly2leg(pol)
            legint = beignet.orthax.legendre.legint(legpol, m=1, k=[i])
            res = beignet.orthax.legendre.leg2poly(legint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            legpol = beignet.orthax.legendre.poly2leg(pol)
            legint = beignet.orthax.legendre.legint(legpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legendre.legval(-1, legint), i
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            legpol = beignet.orthax.legendre.poly2leg(pol)
            legint = beignet.orthax.legendre.legint(legpol, m=1, k=[i], scl=2)
            res = beignet.orthax.legendre.leg2poly(legint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.legendre.legint(tgt, m=1)
                res = beignet.orthax.legendre.legint(pol, m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legendre.legint(tgt, m=1, k=[k])
                res = beignet.orthax.legendre.legint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legendre.legint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.legendre.legint(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legendre.legint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.legendre.legint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_legint_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.legendre.legint(c) for c in c2d.T]).T
        res = beignet.orthax.legendre.legint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legendre.legint(c) for c in c2d])
        res = beignet.orthax.legendre.legint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legendre.legint(c, k=3) for c in c2d])
        res = beignet.orthax.legendre.legint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        numpy.testing.assert_array_equal(
            beignet.orthax.legendre.legint((1, 2, 3), 0), (1, 2, 3)
        )


class TestDerivative:
    def test_legder(self):
        numpy.testing.assert_raises(TypeError, beignet.orthax.legendre.legder, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legendre.legder, [0], -1)

        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.legendre.legder(tgt, m=0)
            numpy.testing.assert_array_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.legendre.legder(
                    beignet.orthax.legendre.legint(tgt, m=j), m=j
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.legendre.legder(
                    beignet.orthax.legendre.legint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.legendre.legder(c) for c in c2d.T]).T
        res = beignet.orthax.legendre.legder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legendre.legder(c) for c in c2d])
        res = beignet.orthax.legendre.legder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legder(c, 4), [0])


class TestVander:
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_legvander(self):
        x = numpy.arange(3)
        v = beignet.orthax.legendre.legvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.legendre.legval(x, coef)
            )

        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.orthax.legendre.legvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.legendre.legval(x, coef)
            )

    def test_legvander2d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.orthax.legendre.legvander2d(x1, x2, (1, 2))
        tgt = beignet.orthax.legendre.legval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.legendre.legvander2d([x1], [x2], (1, 2))
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_legvander3d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.orthax.legendre.legvander3d(x1, x2, x3, (1, 2, 3))
        tgt = beignet.orthax.legendre.legval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.legendre.legvander3d([x1], [x2], [x3], (1, 2, 3))
        numpy.testing.assert_(van.shape == (1, 5, 24))

    def test_legvander_negdeg(self):
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legvander, (1, 2, 3), -1
        )


class TestFitting:
    def test_legfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legfit, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legfit, [1], [1], (-1,)
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legfit, [1], [1], (2, -1, 6)
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.legendre.legfit, [1], [1], ()
        )

        x = numpy.linspace(0, 2)
        y = f(x)

        coef3 = beignet.orthax.legendre.legfit(x, y, 3)
        numpy.testing.assert_array_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef3), y
        )
        coef3 = beignet.orthax.legendre.legfit(x, y, (0, 1, 2, 3))
        numpy.testing.assert_array_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef3), y
        )

        coef4 = beignet.orthax.legendre.legfit(x, y, 4)
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef4), y
        )
        coef4 = beignet.orthax.legendre.legfit(x, y, (0, 1, 2, 3, 4))
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef4), y
        )

        coef4 = beignet.orthax.legendre.legfit(x, y, (2, 3, 4, 1, 0))
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef4), y
        )

        coef2d = beignet.orthax.legendre.legfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.orthax.legendre.legfit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.orthax.legendre.legfit(x, yw, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.orthax.legendre.legfit(x, yw, (0, 1, 2, 3), w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)

        wcoef2d = beignet.orthax.legendre.legfit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.orthax.legendre.legfit(
            x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w
        )
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

        x = [1, 1j, -1, -1j]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legfit(x, x, 1), [0, 1]
        )
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legfit(x, x, (0, 1)), [0, 1]
        )

        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.orthax.legendre.legfit(x, y, 4)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef1), y
        )
        coef2 = beignet.orthax.legendre.legfit(x, y, (0, 2, 4))
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legval(x, coef2), y
        )
        numpy.testing.assert_array_almost_equal(coef1, coef2)


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legcompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legcompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.orthax.legendre.legcompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.orthax.legendre.legcompanion([1, 2])[0, 0] == -0.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.orthax.legendre.leggauss(100)

        v = beignet.orthax.legendre.legvander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

        tgt = 2.0
        numpy.testing.assert_array_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_legfromroots(self):
        res = beignet.orthax.legendre.legfromroots([])
        numpy.testing.assert_array_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            pol = beignet.orthax.legendre.legfromroots(roots)
            res = beignet.orthax.legendre.legval(roots, pol)
            tgt = 0
            numpy.testing.assert_(len(pol) == i + 1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legendre.leg2poly(pol)[-1], 1
            )
            numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legroots(self):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legroots([1]), []
        )
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legendre.legroots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.orthax.legendre.legroots(
                beignet.orthax.legendre.legfromroots(tgt)
            )
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_legtrim(self):
        coef = [2, -1, 1, 0]

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legendre.legtrim, coef, -1
        )

        numpy.testing.assert_array_equal(
            beignet.orthax.legendre.legtrim(coef), coef[:-1]
        )
        numpy.testing.assert_array_equal(
            beignet.orthax.legendre.legtrim(coef, 1), coef[:-3]
        )
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legtrim(coef, 2), [0])

    def test_legline(self):
        numpy.testing.assert_array_equal(beignet.orthax.legendre.legline(3, 4), [3, 4])

    def test_legline_zeroscl(self):
        numpy.testing.assert_array_equal(
            trim(beignet.orthax.legendre.legline(3, 0)), [3]
        )

    def test_leg2poly(self):
        for i in range(10):
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legendre.leg2poly([0] * i + [1]), Llist[i]
            )

    def test_poly2leg(self):
        for i in range(10):
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legendre.poly2leg(Llist[i]), [0] * i + [1]
            )

    def test_weight(self):
        x = numpy.linspace(-1, 1, 11)
        tgt = 1.0
        res = beignet.orthax.legendre.legweight(x)
        numpy.testing.assert_array_almost_equal(res, tgt)
