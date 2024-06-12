import copy
import functools
import pickle

import beignet.polynomial.polynomial
import numpy
import numpy.testing


def trim(x):
    return beignet.polynomial.polynomial.polytrim(x, tol=1e-6)


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
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polydomain, [-1, 1])

    def test_polyzero(self):
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polyzero, [0])

    def test_polyone(self):
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polyone, [1])

    def test_polyx(self):
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polyx, [0, 1])

    def test_copy(self):
        x = beignet.polynomial.polynomial.Polynomial([1, 2, 3])
        y = copy.deepcopy(x)
        numpy.testing.assert_equal(x, y)

    def test_pickle(self):
        x = beignet.polynomial.polynomial.Polynomial([1, 2, 3])
        y = pickle.loads(pickle.dumps(x))
        numpy.testing.assert_equal(x, y)


class TestArithmetic:
    def test_polyadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.polynomial.polynomial.polyadd(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polysub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.polynomial.polynomial.polysub(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polymulx(self):
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polymulx([0]), [0])
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polymulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i + 1) + [1]
            numpy.testing.assert_equal(beignet.polynomial.polynomial.polymulx(ser), tgt)

    def test_polymul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(i + j + 1)
                tgt[i + j] += 1
                res = beignet.polynomial.polynomial.polymul(
                    [0] * i + [1], [0] * j + [1]
                )
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polydiv(self):
        # check zero division
        numpy.testing.assert_raises(
            ZeroDivisionError, beignet.polynomial.polynomial.polydiv, [1], [0]
        )

        # check scalar division
        quo, rem = beignet.polynomial.polynomial.polydiv([2], [2])
        numpy.testing.assert_equal((quo, rem), (1, 0))
        quo, rem = beignet.polynomial.polynomial.polydiv([2, 2], [2])
        numpy.testing.assert_equal((quo, rem), ((1, 1), 0))

        # check rest.
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1, 2]
                cj = [0] * j + [1, 2]
                tgt = beignet.polynomial.polynomial.polyadd(ci, cj)
                quo, rem = beignet.polynomial.polynomial.polydiv(tgt, ci)
                res = beignet.polynomial.polynomial.polyadd(
                    beignet.polynomial.polynomial.polymul(quo, ci), rem
                )
                numpy.testing.assert_equal(res, tgt, err_msg=msg)

    def test_polypow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.polynomial.polynomial.polymul, [c] * j, numpy.array([1])
                )
                res = beignet.polynomial.polynomial.polypow(c, j)
                numpy.testing.assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([1.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_polyval(self):
        # check empty input
        numpy.testing.assert_equal(
            beignet.polynomial.polynomial.polyval([], [1]).size, 0
        )

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(5):
            tgt = y[i]
            res = beignet.polynomial.polynomial.polyval(x, [0] * i + [1])
            numpy.testing.assert_almost_equal(res, tgt)
        tgt = x * (x**2 - 1)
        res = beignet.polynomial.polynomial.polyval(x, [0, -1, 0, 1])
        numpy.testing.assert_almost_equal(res, tgt)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial.polynomial.polyval(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial.polynomial.polyval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial.polynomial.polyval(x, [1, 0, 0]).shape, dims
            )

        # check masked arrays are processed correctly
        mask = [False, True, False]
        mx = numpy.ma.array([1, 2, 3], mask=mask)
        res = numpy.polyval([7, 5, 3], mx)
        numpy.testing.assert_array_equal(res.mask, mask)

        # check subtypes of ndarray are preserved
        class C(numpy.ndarray):
            pass

        cx = numpy.array([1, 2, 3]).view(C)
        numpy.testing.assert_equal(type(numpy.polyval([2, 3, 4], cx)), C)

    def test_polyvalfromroots(self):
        # check exception for broadcasting x values over root array with
        # too few dimensions
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial.polynomial.polyvalfromroots,
            [1],
            [1],
            tensor=False,
        )

        # check empty input
        numpy.testing.assert_equal(
            beignet.polynomial.polynomial.polyvalfromroots([], [1]).size, 0
        )
        numpy.testing.assert_(
            beignet.polynomial.polynomial.polyvalfromroots([], [1]).shape == (0,)
        )

        # check empty input + multidimensional roots
        numpy.testing.assert_equal(
            beignet.polynomial.polynomial.polyvalfromroots([], [[1] * 5]).size, 0
        )
        numpy.testing.assert_(
            beignet.polynomial.polynomial.polyvalfromroots([], [[1] * 5]).shape
            == (5, 0)
        )

        # check scalar input
        numpy.testing.assert_equal(
            beignet.polynomial.polynomial.polyvalfromroots(1, 1), 0
        )
        numpy.testing.assert_(
            beignet.polynomial.polynomial.polyvalfromroots(1, numpy.ones((3, 3))).shape
            == (3,)
        )

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(1, 5):
            tgt = y[i]
            res = beignet.polynomial.polynomial.polyvalfromroots(x, [0] * i)
            numpy.testing.assert_almost_equal(res, tgt)
        tgt = x * (x - 1) * (x + 1)
        res = beignet.polynomial.polynomial.polyvalfromroots(x, [-1, 0, 1])
        numpy.testing.assert_almost_equal(res, tgt)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.polynomial.polynomial.polyvalfromroots(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial.polynomial.polyvalfromroots(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.polynomial.polynomial.polyvalfromroots(x, [1, 0, 0]).shape, dims
            )

        # check compatibility with factorization
        ptest = [15, 2, -16, -2, 1]
        r = beignet.polynomial.polynomial.polyroots(ptest)
        x = numpy.linspace(-1, 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, ptest),
            beignet.polynomial.polynomial.polyvalfromroots(x, r),
        )

        # check multidimensional arrays of roots and values
        # check tensor=False
        rshape = (3, 5)
        x = numpy.arange(-3, 2)
        r = numpy.random.randint(-5, 5, size=rshape)
        res = beignet.polynomial.polynomial.polyvalfromroots(x, r, tensor=False)
        tgt = numpy.empty(r.shape[1:])
        for ii in range(tgt.size):
            tgt[ii] = beignet.polynomial.polynomial.polyvalfromroots(x[ii], r[:, ii])
        numpy.testing.assert_equal(res, tgt)

        # check tensor=True
        x = numpy.vstack([x, 2 * x])
        res = beignet.polynomial.polynomial.polyvalfromroots(x, r, tensor=True)
        tgt = numpy.empty(r.shape[1:] + x.shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0]):
                tgt[ii, jj, :] = beignet.polynomial.polynomial.polyvalfromroots(
                    x[jj], r[:, ii]
                )
        numpy.testing.assert_equal(res, tgt)

    def test_polyval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises_regex(
            ValueError,
            "incompatible",
            beignet.polynomial.polynomial.polyval2d,
            x1,
            x2[:2],
            self.c2d,
        )

        # test values
        tgt = y1 * y2
        res = beignet.polynomial.polynomial.polyval2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.polynomial.polyval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_polyval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises_regex(
            ValueError,
            "incompatible",
            beignet.polynomial.polynomial.polyval3d,
            x1,
            x2,
            x3[:2],
            self.c3d,
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.polynomial.polynomial.polyval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.polynomial.polyval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_polygrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.polynomial.polynomial.polygrid2d(x1, x2, self.c2d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.polynomial.polygrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_polygrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.polynomial.polynomial.polygrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.polynomial.polynomial.polygrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_polyint(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyint, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyint, [0], -1
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyint, [0], 1, [0, 0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyint, [0], lbnd=[0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyint, [0], scl=[0]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyint, [0], axis=0.5
        )
        with numpy.testing.assert_warns(DeprecationWarning):
            beignet.polynomial.polynomial.polyint([1, 1], 1.0)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.polynomial.polynomial.polyint([0], m=i, k=k)
            numpy.testing.assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            res = beignet.polynomial.polynomial.polyint(pol, m=1, k=[i])
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            res = beignet.polynomial.polynomial.polyint(pol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.polynomial.polyval(-1, res), i
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            res = beignet.polynomial.polynomial.polyint(pol, m=1, k=[i], scl=2)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.polynomial.polynomial.polyint(tgt, m=1)
                res = beignet.polynomial.polynomial.polyint(pol, m=j)
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial.polynomial.polyint(tgt, m=1, k=[k])
                res = beignet.polynomial.polynomial.polyint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial.polynomial.polyint(
                        tgt, m=1, k=[k], lbnd=-1
                    )
                res = beignet.polynomial.polynomial.polyint(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.polynomial.polynomial.polyint(tgt, m=1, k=[k], scl=2)
                res = beignet.polynomial.polynomial.polyint(
                    pol, m=j, k=list(range(j)), scl=2
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_polyint_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial.polynomial.polyint(c) for c in c2d.T]).T
        res = beignet.polynomial.polynomial.polyint(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial.polynomial.polyint(c) for c in c2d])
        res = beignet.polynomial.polynomial.polyint(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial.polynomial.polyint(c, k=3) for c in c2d])
        res = beignet.polynomial.polynomial.polyint(c2d, k=3, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)


class TestDerivative:
    def test_polyder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyder, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyder, [0], -1
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.polynomial.polyder(tgt, m=0)
            numpy.testing.assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial.polynomial.polyder(
                    beignet.polynomial.polynomial.polyint(tgt, m=j), m=j
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.polynomial.polynomial.polyder(
                    beignet.polynomial.polynomial.polyint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_polyder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.polynomial.polynomial.polyder(c) for c in c2d.T]).T
        res = beignet.polynomial.polynomial.polyder(c2d, axis=0)
        numpy.testing.assert_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.polynomial.polynomial.polyder(c) for c in c2d])
        res = beignet.polynomial.polynomial.polyder(c2d, axis=1)
        numpy.testing.assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_polyvander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.polynomial.polynomial.polyvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial.polynomial.polyval(x, coef)
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.polynomial.polynomial.polyvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                v[..., i], beignet.polynomial.polynomial.polyval(x, coef)
            )

    def test_polyvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.polynomial.polynomial.polyvander2d(x1, x2, [1, 2])
        tgt = beignet.polynomial.polynomial.polyval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial.polynomial.polyvander2d([x1], [x2], [1, 2])
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_polyvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.polynomial.polynomial.polyvander3d(x1, x2, x3, [1, 2, 3])
        tgt = beignet.polynomial.polynomial.polyval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_almost_equal(res, tgt)

        # check shape
        van = beignet.polynomial.polynomial.polyvander3d([x1], [x2], [x3], [1, 2, 3])
        numpy.testing.assert_(van.shape == (1, 5, 24))

    def test_polyvandernegdeg(self):
        x = numpy.arange(3)
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyvander, x, -1
        )


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polycompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polycompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.polynomial.polynomial.polycompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.polynomial.polynomial.polycompanion([1, 2])[0, 0] == -0.5
        )


class TestMisc:
    def test_polyfromroots(self):
        res = beignet.polynomial.polynomial.polyfromroots([])
        numpy.testing.assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            tgt = Tlist[i]
            res = beignet.polynomial.polynomial.polyfromroots(roots) * 2 ** (i - 1)
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_polyroots(self):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyroots([1]), []
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyroots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.polynomial.polynomial.polyroots(
                beignet.polynomial.polynomial.polyfromroots(tgt)
            )
            numpy.testing.assert_almost_equal(trim(res), trim(tgt))

    def test_polyfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyfit, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError,
            beignet.polynomial.polynomial.polyfit,
            [1],
            [1],
            [
                -1,
            ],
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polyfit, [1], [1], [2, -1, 6]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.polynomial.polynomial.polyfit, [1], [1], []
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.polynomial.polynomial.polyfit(x, y, 3)
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, coef3), y
        )
        coef3 = beignet.polynomial.polynomial.polyfit(x, y, [0, 1, 2, 3])
        numpy.testing.assert_equal(len(coef3), 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, coef3), y
        )
        #
        coef4 = beignet.polynomial.polynomial.polyfit(x, y, 4)
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, coef4), y
        )
        coef4 = beignet.polynomial.polynomial.polyfit(x, y, [0, 1, 2, 3, 4])
        numpy.testing.assert_equal(len(coef4), 5)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, coef4), y
        )
        #
        coef2d = beignet.polynomial.polynomial.polyfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.polynomial.polynomial.polyfit(
            x, numpy.array([y, y]).T, [0, 1, 2, 3]
        )
        numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        wcoef3 = beignet.polynomial.polynomial.polyfit(x, yw, 3, w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.polynomial.polynomial.polyfit(x, yw, [0, 1, 2, 3], w=w)
        numpy.testing.assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.polynomial.polynomial.polyfit(
            x, numpy.array([yw, yw]).T, 3, w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.polynomial.polynomial.polyfit(
            x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
        )
        numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyfit(x, x, 1), [0, 1]
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyfit(x, x, [0, 1]), [0, 1]
        )
        # test fitting only even Polyendre polynomials
        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.polynomial.polynomial.polyfit(x, y, 4)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, coef1), y
        )
        coef2 = beignet.polynomial.polynomial.polyfit(x, y, [0, 2, 4])
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polynomial.polyval(x, coef2), y
        )
        numpy.testing.assert_almost_equal(coef1, coef2)

    def test_polytrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polynomial.polytrim, coef, -1
        )

        # Test results
        numpy.testing.assert_equal(
            beignet.polynomial.polynomial.polytrim(coef), coef[:-1]
        )
        numpy.testing.assert_equal(
            beignet.polynomial.polynomial.polytrim(coef, 1), coef[:-3]
        )
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polytrim(coef, 2), [0])

    def test_polyline(self):
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polyline(3, 4), [3, 4])

    def test_polyline_zero(self):
        numpy.testing.assert_equal(beignet.polynomial.polynomial.polyline(3, 0), [3])
