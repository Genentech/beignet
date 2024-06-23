import functools

import beignet.orthax
import numpy
import numpy.testing


def trim(x):
    return beignet.orthax.polytrim(x, tol=1e-6)


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


class TestConstants:
    def test_polydomain(self):
        numpy.testing.assert_equal(beignet.orthax.polydomain, numpy.array([-1, 1]))

    def test_polyzero(self):
        numpy.testing.assert_equal(beignet.orthax.polyzero, numpy.array([0]))

    def test_polyone(self):
        numpy.testing.assert_equal(beignet.orthax.polyone, numpy.array([1]))

    def test_polyx(self):
        numpy.testing.assert_equal(beignet.orthax.polyx, numpy.array([0, 1]))


class TestArithmetic:
    def test_polyadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.orthax.polyadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polysub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.polysub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polymulx(self):
        numpy.testing.assert_array_equal(beignet.orthax.polymulx([0]), [0, 0])
        numpy.testing.assert_array_equal(beignet.orthax.polymulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i + 1) + [1]
            numpy.testing.assert_array_equal(beignet.orthax.polymulx(ser), tgt)

    def test_polymul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(i + j + 1)
                tgt[i + j] += 1
                res = beignet.orthax.polymul([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polydiv(self):
        quo, rem = beignet.orthax.polydiv([2], [2])
        numpy.testing.assert_array_equal(quo, [1])
        numpy.testing.assert_array_equal(rem, [0])
        quo, rem = beignet.orthax.polydiv([2, 2], [2])
        numpy.testing.assert_array_equal(quo, (1, 1))
        numpy.testing.assert_array_equal(rem, [0])

        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0.0] * i + [1.0, 2.0]
                cj = [0.0] * j + [1.0, 2.0]
                tgt = beignet.orthax.polyadd(ci, cj)
                quo, rem = beignet.orthax.polydiv(tgt, ci)
                res = beignet.orthax.polyadd(beignet.orthax.polymul(quo, ci), rem)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polypow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.orthax.polymul, [c] * j, numpy.array([1])
                )
                res = beignet.orthax.polypow(c, j)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    c1d = numpy.array([1.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    def test_polyval(self):
        numpy.testing.assert_equal(beignet.orthax.polyval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(5):
            tgt = y[i]
            res = beignet.orthax.polyval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt)
        tgt = x * (x**2 - 1)
        res = beignet.orthax.polyval(x, [0, -1, 0, 1])
        numpy.testing.assert_array_almost_equal(res, tgt)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(beignet.orthax.polyval(x, [1]).shape, dims)
            numpy.testing.assert_equal(beignet.orthax.polyval(x, [1, 0]).shape, dims)
            numpy.testing.assert_equal(beignet.orthax.polyval(x, [1, 0, 0]).shape, dims)

        mask = [False, True, False]
        mx = numpy.ma.array([1, 2, 3], mask=mask)
        res = numpy.polyval([7, 5, 3], mx)
        numpy.testing.assert_array_equal(res.mask, mask)

    def test_polyvalfromroots(self):
        numpy.testing.assert_raises(
            ValueError,
            beignet.orthax.polyvalfromroots,
            [1],
            [1],
            tensor=False,
        )

        numpy.testing.assert_equal(beignet.orthax.polyvalfromroots([], [1]).size, 0)
        numpy.testing.assert_(beignet.orthax.polyvalfromroots([], [1]).shape == (0,))

        numpy.testing.assert_equal(
            beignet.orthax.polyvalfromroots([], [[1] * 5]).size, 0
        )
        numpy.testing.assert_(
            beignet.orthax.polyvalfromroots([], [[1] * 5]).shape == (5, 0)
        )

        numpy.testing.assert_array_equal(beignet.orthax.polyvalfromroots(1, 1), 0)
        numpy.testing.assert_(
            beignet.orthax.polyvalfromroots(1, numpy.ones((3, 3))).shape == (3,)
        )

        x = numpy.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(1, 5):
            tgt = y[i]
            res = beignet.orthax.polyvalfromroots(x, [0] * i)
            numpy.testing.assert_array_almost_equal(res, tgt)
        tgt = x * (x - 1) * (x + 1)
        res = beignet.orthax.polyvalfromroots(x, [-1, 0, 1])
        numpy.testing.assert_array_almost_equal(res, tgt)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.orthax.polyvalfromroots(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.orthax.polyvalfromroots(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.orthax.polyvalfromroots(x, [1, 0, 0]).shape, dims
            )

        ptest = [15, 2, -16, -2, 1]
        r = beignet.orthax.polyroots(ptest)
        x = numpy.linspace(-1, 1)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polyval(x, ptest),
            beignet.orthax.polyvalfromroots(x, r),
        )

        rshape = (3, 5)
        x = numpy.arange(-3, 2)
        r = numpy.random.randint(-5, 5, size=rshape)
        res = beignet.orthax.polyvalfromroots(x, r, tensor=False)
        tgt = numpy.empty(r.shape[1:])
        for ii in range(tgt.size):
            tgt[ii] = beignet.orthax.polyvalfromroots(x[ii], r[:, ii])
        numpy.testing.assert_array_equal(res, tgt)

        x = numpy.vstack([x, 2 * x])
        res = beignet.orthax.polyvalfromroots(x, r, tensor=True)
        tgt = numpy.empty(r.shape[1:] + x.shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0]):
                tgt[ii, jj, :] = beignet.orthax.polyvalfromroots(x[jj], r[:, ii])
        numpy.testing.assert_array_equal(res, tgt)

    def test_polyval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises_regex(
            ValueError,
            "incompatible",
            beignet.orthax.polyval2d,
            x1,
            x2[:2],
            self.c2d,
        )

        tgt = y1 * y2
        res = beignet.orthax.polyval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polyval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_polyval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises_regex(
            ValueError,
            "incompatible",
            beignet.orthax.polyval3d,
            x1,
            x2,
            x3[:2],
            self.c3d,
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.polyval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polyval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_polygrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.polygrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polygrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_polygrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.polygrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polygrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_polyint(self):  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], -1)
        numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], 1, [0, 0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], lbnd=[0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], scl=[0])
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyint, [0], axis=0.5)

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.polyint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(trim(res), [0, 1])

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            res = beignet.orthax.polyint(pol, m=1, k=[i])
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            res = beignet.orthax.polyint(pol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(-1, res), i)

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            res = beignet.orthax.polyint(pol, m=1, k=[i], scl=2)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.polyint(tgt, m=1)
                res = beignet.orthax.polyint(pol, m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.polyint(tgt, m=1, k=[k])
                res = beignet.orthax.polyint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.polyint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.polyint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.polyint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.polyint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_polyint_axis(self):
        c2d = numpy.random.random((3, 6))

        tgt = numpy.vstack([beignet.orthax.polyint(c) for c in c2d.T]).T
        res = beignet.orthax.polyint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.polyint(c) for c in c2d])
        res = beignet.orthax.polyint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.polyint(c, k=3) for c in c2d])
        res = beignet.orthax.polyint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestDerivative:
    def test_polyder(self):
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyder, [0], 0.5)

        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.polyder(tgt, m=0)
            numpy.testing.assert_array_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.polyder(beignet.orthax.polyint(tgt, m=j), m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.polyder(
                    beignet.orthax.polyint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_polyder_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.polyder(c) for c in c2d.T]).T
        res = beignet.orthax.polyder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.polyder(c) for c in c2d])
        res = beignet.orthax.polyder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestVander:
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_polyvander(self):
        x = numpy.arange(3)
        v = beignet.orthax.polyvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.polyval(x, coef)
            )

        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.orthax.polyvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.polyval(x, coef)
            )

    def test_polyvander2d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.orthax.polyvander2d(x1, x2, (1, 2))
        tgt = beignet.orthax.polyval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.polyvander2d([x1], [x2], (1, 2))
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_polyvander3d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.orthax.polyvander3d(x1, x2, x3, (1, 2, 3))
        tgt = beignet.orthax.polyval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.polyvander3d([x1], [x2], [x3], (1, 2, 3))
        numpy.testing.assert_(van.shape == (1, 5, 24))

    def test_polyvandernegdeg(self):
        x = numpy.arange(3)
        numpy.testing.assert_raises(ValueError, beignet.orthax.polyvander, x, -1)


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(ValueError, beignet.orthax.polycompanion, [])
        numpy.testing.assert_raises(ValueError, beignet.orthax.polycompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(beignet.orthax.polycompanion(coef).shape == (i, i))

    def test_linear_root(self):
        numpy.testing.assert_(beignet.orthax.polycompanion([1, 2])[0, 0] == -0.5)


class TestMisc:
    def test_polyfromroots(self):
        res = beignet.orthax.polyfromroots([])
        numpy.testing.assert_array_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            tgt = Tlist[i]
            res = beignet.orthax.polyfromroots(roots) * 2 ** (i - 1)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_polyroots(self):
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyroots([1]), [])
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polyroots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.orthax.polyroots(beignet.orthax.polyfromroots(tgt))
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_polyfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        numpy.testing.assert_raises(ValueError, beignet.orthax.polyfit, [1], [1], -1)
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [[1]], [1], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [], [1], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [1], [[[1]]], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [1, 2], [1], 0)
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [1], [1, 2], 0)
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.polyfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.polyfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(ValueError, beignet.orthax.polyfit, [1], [1], (-1,))
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.polyfit, [1], [1], (2, -1, 6)
        )
        numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [1], [1], ())

        x = numpy.linspace(0, 2)
        y = f(x)

        coef3 = beignet.orthax.polyfit(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(x, coef3), y)
        coef3 = beignet.orthax.polyfit(x, y, (0, 1, 2, 3))
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(x, coef3), y)

        coef4 = beignet.orthax.polyfit(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(x, coef4), y)
        coef4 = beignet.orthax.polyfit(x, y, (0, 1, 2, 3, 4))
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(x, coef4), y)

        coef2d = beignet.orthax.polyfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.orthax.polyfit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        wcoef3 = beignet.orthax.polyfit(x, yw, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.orthax.polyfit(x, yw, (0, 1, 2, 3), w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)

        wcoef2d = beignet.orthax.polyfit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.orthax.polyfit(x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w)
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

        x = [1, 1j, -1, -1j]
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyfit(x, x, 1), [0, 1])
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polyfit(x, x, (0, 1)), [0, 1]
        )

        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.orthax.polyfit(x, y, 4)
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(x, coef1), y)
        coef2 = beignet.orthax.polyfit(x, y, (0, 2, 4))
        numpy.testing.assert_array_almost_equal(beignet.orthax.polyval(x, coef2), y)
        numpy.testing.assert_array_almost_equal(coef1, coef2)

    def test_polytrim(self):
        coef = [2, -1, 1, 0]

        numpy.testing.assert_raises(ValueError, beignet.orthax.polytrim, coef, -1)

        numpy.testing.assert_array_equal(beignet.orthax.polytrim(coef), coef[:-1])
        numpy.testing.assert_array_equal(beignet.orthax.polytrim(coef, 1), coef[:-3])
        numpy.testing.assert_array_equal(beignet.orthax.polytrim(coef, 2), [0])

    def test_polyline(self):
        numpy.testing.assert_array_equal(beignet.orthax.polyline(3, 4), [3, 4])

    def test_polyline_zero(self):
        numpy.testing.assert_array_equal(beignet.orthax.polyline(3, 0), [3, 0])
