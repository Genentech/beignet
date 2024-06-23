import functools

import beignet.orthax
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
    return beignet.orthax.hermetrim(x, tol=1e-6)


class TestConstants:
    def test_hermedomain(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermedomain, [-1, 1])

    def test_hermezero(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermezero, [0])

    def test_hermeone(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermeone, [1])

    def test_hermex(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermex, [0, 1])


class TestArithmetic:
    x = numpy.linspace(-3, 3, 100)

    def test_hermeadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.orthax.hermeadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermesub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.hermesub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermemulx(self):
        numpy.testing.assert_array_equal(trim(beignet.orthax.hermemulx([0])), [0])
        numpy.testing.assert_array_equal(trim(beignet.orthax.hermemulx([1])), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i, 0, 1]
            numpy.testing.assert_array_equal(trim(beignet.orthax.hermemulx(ser)), tgt)

    def test_hermemul(self):
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.orthax.hermeval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = beignet.orthax.hermeval(self.x, pol2)
                pol3 = beignet.orthax.hermemul(pol1, pol2)
                val3 = beignet.orthax.hermeval(self.x, pol3)
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_hermediv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.orthax.hermeadd(ci, cj)
                quo, rem = beignet.orthax.hermediv(tgt, ci)
                res = beignet.orthax.hermeadd(beignet.orthax.hermemul(quo, ci), rem)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermepow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.orthax.hermemul, [c] * j, numpy.array([1])
                )
                res = beignet.orthax.hermepow(c, j)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_hermeval(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermeval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in Helist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.hermeval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.hermeval(x, [1]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.hermeval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.hermeval(x, [1, 0, 0]).shape, dims
            )

    def test_hermeval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermeval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.hermeval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermeval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermeval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermeval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.hermeval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermeval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermegrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.hermegrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermegrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_hermegrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.hermegrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermegrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_hermeint(self):  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermeint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], -1)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], 1, [0, 0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], lbnd=[0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], scl=[0])
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermeint, [0], axis=0.5)

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.hermeint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(trim(res), [0, 1])

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermepol = beignet.orthax.poly2herme(pol)
            hermeint = beignet.orthax.hermeint(hermepol, m=1, k=[i])
            res = beignet.orthax.herme2poly(hermeint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            hermepol = beignet.orthax.poly2herme(pol)
            hermeint = beignet.orthax.hermeint(hermepol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermeval(-1, hermeint), i
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            hermepol = beignet.orthax.poly2herme(pol)
            hermeint = beignet.orthax.hermeint(hermepol, m=1, k=[i], scl=2)
            res = beignet.orthax.herme2poly(hermeint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1)
                res = beignet.orthax.hermeint(pol, m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1, k=[k])
                res = beignet.orthax.hermeint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.hermeint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_hermeint_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.hermeint(c) for c in c2d.T]).T
        res = beignet.orthax.hermeint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.hermeint(c) for c in c2d])
        res = beignet.orthax.hermeint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.hermeint(c, k=3) for c in c2d])
        res = beignet.orthax.hermeint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestDerivative:
    def test_hermeder(self):
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermeder, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeder, [0], -1)

        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.hermeder(tgt, m=0)
            numpy.testing.assert_array_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.hermeder(beignet.orthax.hermeint(tgt, m=j), m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.hermeder(
                    beignet.orthax.hermeint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_hermeder_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.hermeder(c) for c in c2d.T]).T
        res = beignet.orthax.hermeder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.hermeder(c) for c in c2d])
        res = beignet.orthax.hermeder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestVander:
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_hermevander(self):
        x = numpy.arange(3)
        v = beignet.orthax.hermevander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.hermeval(x, coef)
            )

        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.orthax.hermevander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.hermeval(x, coef)
            )

    def test_hermevander2d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.orthax.hermevander2d(x1, x2, (1, 2))
        tgt = beignet.orthax.hermeval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.hermevander2d([x1], [x2], (1, 2))
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_hermevander3d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.orthax.hermevander3d(x1, x2, x3, (1, 2, 3))
        tgt = beignet.orthax.hermeval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.hermevander3d([x1], [x2], [x3], (1, 2, 3))
        numpy.testing.assert_(van.shape == (1, 5, 24))


class TestFitting:
    def test_hermefit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        numpy.testing.assert_raises(ValueError, beignet.orthax.hermefit, [1], [1], -1)
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermefit, [[1]], [1], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermefit, [], [1], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermefit, [1], [[[1]]], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermefit, [1, 2], [1], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermefit, [1], [1, 2], 0)
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.hermefit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.hermefit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermefit, [1], [1], (-1,)
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermefit, [1], [1], (2, -1, 6)
        )
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermefit, [1], [1], ())

        x = numpy.linspace(0, 2)
        y = f(x)

        coef3 = beignet.orthax.hermefit(x, y, 3)
        numpy.testing.assert_array_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef3), y)
        coef3 = beignet.orthax.hermefit(x, y, (0, 1, 2, 3))
        numpy.testing.assert_array_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef3), y)

        coef4 = beignet.orthax.hermefit(x, y, 4)
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef4), y)
        coef4 = beignet.orthax.hermefit(x, y, (0, 1, 2, 3, 4))
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef4), y)

        coef4 = beignet.orthax.hermefit(x, y, (2, 3, 4, 1, 0))
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef4), y)

        coef2d = beignet.orthax.hermefit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.orthax.hermefit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.orthax.hermefit(x, yw, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.orthax.hermefit(x, yw, (0, 1, 2, 3), w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)

        wcoef2d = beignet.orthax.hermefit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.orthax.hermefit(x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w)
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

        x = [1, 1j, -1, -1j]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.hermefit(x, x, 1), [0, 1]
        )
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.hermefit(x, x, (0, 1)), [0, 1]
        )

        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.orthax.hermefit(x, y, 4)
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef1), y)
        coef2 = beignet.orthax.hermefit(x, y, (0, 2, 4))
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeval(x, coef2), y)
        numpy.testing.assert_array_almost_equal(coef1, coef2)


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermecompanion, [])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermecompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(beignet.orthax.hermecompanion(coef).shape == (i, i))

    def test_linear_root(self):
        numpy.testing.assert_(beignet.orthax.hermecompanion([1, 2])[0, 0] == -0.5)


class TestGauss:
    def test_100(self):
        x, w = beignet.orthax.hermegauss(100)

        v = beignet.orthax.hermevander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

        tgt = numpy.sqrt(2 * numpy.pi)
        numpy.testing.assert_array_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_hermefromroots(self):
        res = beignet.orthax.hermefromroots([])
        numpy.testing.assert_array_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            pol = beignet.orthax.hermefromroots(roots)
            res = beignet.orthax.hermeval(roots, pol)
            tgt = 0
            numpy.testing.assert_(len(pol) == i + 1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.herme2poly(pol)[-1], 1
            )
            numpy.testing.assert_array_almost_equal(res, tgt)

    def test_hermeroots(self):
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeroots([1]), [])
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermeroots([1, 1]), [-1])
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.orthax.hermeroots(beignet.orthax.hermefromroots(tgt))
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_hermetrim(self):
        coef = [2, -1, 1, 0]

        numpy.testing.assert_raises(ValueError, beignet.orthax.hermetrim, coef, -1)

        numpy.testing.assert_array_equal(beignet.orthax.hermetrim(coef), coef[:-1])
        numpy.testing.assert_array_equal(beignet.orthax.hermetrim(coef, 1), coef[:-3])
        numpy.testing.assert_array_equal(beignet.orthax.hermetrim(coef, 2), [0])

    def test_hermeline(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermeline(3, 4), [3, 4])

    def test_herme2poly(self):
        for i in range(10):
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.herme2poly([0] * i + [1]), Helist[i]
            )

    def test_poly2herme(self):
        for i in range(10):
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.poly2herme(Helist[i]), [0] * i + [1]
            )

    def test_weight(self):
        x = numpy.linspace(-5, 5, 11)
        tgt = numpy.exp(-0.5 * x**2)
        res = beignet.orthax.hermeweight(x)
        numpy.testing.assert_array_almost_equal(res, tgt)
