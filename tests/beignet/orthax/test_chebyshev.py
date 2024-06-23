import functools

import beignet.orthax.chebyshev
import numpy
import numpy.polynomial.polynomial
import numpy.testing


def trim(x):
    return beignet.orthax.chebyshev.chebtrim(x, tol=1e-6)


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
            res = beignet.orthax.chebyshev._cseries_to_zseries(inp)
            numpy.testing.assert_array_equal(res, tgt)

    def test__zseries_to_cseries(self):
        for i in range(5):
            inp = numpy.array([0.5] * i + [2] + [0.5] * i, numpy.double)
            tgt = numpy.array([2] + [1] * i, numpy.double)
            res = beignet.orthax.chebyshev._zseries_to_cseries(inp)
            numpy.testing.assert_array_equal(res, tgt)


class TestConstants:
    def test_chebdomain(self):
        numpy.testing.assert_array_equal(beignet.orthax.chebyshev.chebdomain, [-1, 1])

    def test_chebzero(self):
        numpy.testing.assert_array_equal(beignet.orthax.chebyshev.chebzero, [0])

    def test_chebone(self):
        numpy.testing.assert_array_equal(beignet.orthax.chebyshev.chebone, [1])

    def test_chebx(self):
        numpy.testing.assert_array_equal(beignet.orthax.chebyshev.chebx, [0, 1])


class TestArithmetic:
    def test_chebadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.orthax.chebyshev.chebadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.chebyshev.chebsub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebmulx(self):
        numpy.testing.assert_array_equal(
            trim(beignet.orthax.chebyshev.chebmulx([0])), [0]
        )
        numpy.testing.assert_array_equal(
            trim(beignet.orthax.chebyshev.chebmulx([1])), [0, 1]
        )
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [0.5, 0, 0.5]
            numpy.testing.assert_array_equal(
                trim(beignet.orthax.chebyshev.chebmulx(ser)), tgt
            )

    def test_chebmul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(i + j + 1)
                tgt[i + j] += 0.5
                tgt[abs(i - j)] += 0.5
                res = beignet.orthax.chebyshev.chebmul([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.orthax.chebyshev.chebadd(ci, cj)
                quo, rem = beignet.orthax.chebyshev.chebdiv(tgt, ci)
                res = beignet.orthax.chebyshev.chebadd(
                    beignet.orthax.chebyshev.chebmul(quo, ci), rem
                )
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(
                    beignet.orthax.chebyshev.chebmul, [c] * j, numpy.array([1])
                )
                res = beignet.orthax.chebyshev.chebpow(c, j)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_chebval(self):
        # check empty input
        numpy.testing.assert_array_equal(
            beignet.orthax.chebyshev.chebval([], [1]).size, 0
        )

        # check normal input)
        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in Tlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.chebyshev.chebval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.chebyshev.chebval(x, [1]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.chebyshev.chebval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.chebyshev.chebval(x, [1, 0, 0]).shape, dims
            )

    def test_chebval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebval2d, x1, x2[:2], self.c2d
        )

        # test values
        tgt = y1 * y2
        res = beignet.orthax.chebyshev.chebval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.orthax.chebyshev.chebval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_chebval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebval3d, x1, x2, x3[:2], self.c3d
        )

        # test values
        tgt = y1 * y2 * y3
        res = beignet.orthax.chebyshev.chebval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.orthax.chebyshev.chebval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_chebgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.chebyshev.chebgrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.orthax.chebyshev.chebgrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_chebgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.chebyshev.chebgrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        # test shape
        z = numpy.ones((2, 3))
        res = beignet.orthax.chebyshev.chebgrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_chebint(self):  # noqa:C901
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebint, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebint, [0], -1
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebint, [0], 1, [0, 0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebint, [0], lbnd=[0]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebint, [0], scl=[0]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebint, [0], axis=0.5
        )

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.chebyshev.chebint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(trim(res), [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            chebpol = beignet.orthax.chebyshev.poly2cheb(pol)
            chebint = beignet.orthax.chebyshev.chebint(chebpol, m=1, k=[i])
            res = beignet.orthax.chebyshev.cheb2poly(chebint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            chebpol = beignet.orthax.chebyshev.poly2cheb(pol)
            chebint = beignet.orthax.chebyshev.chebint(chebpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebyshev.chebval(-1, chebint), i
            )

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            chebpol = beignet.orthax.chebyshev.poly2cheb(pol)
            chebint = beignet.orthax.chebyshev.chebint(chebpol, m=1, k=[i], scl=2)
            res = beignet.orthax.chebyshev.cheb2poly(chebint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.chebyshev.chebint(tgt, m=1)
                res = beignet.orthax.chebyshev.chebint(pol, m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.chebyshev.chebint(tgt, m=1, k=[k])
                res = beignet.orthax.chebyshev.chebint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.chebyshev.chebint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.chebyshev.chebint(
                    pol, m=j, k=list(range(j)), lbnd=-1
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.chebyshev.chebint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.chebyshev.chebint(
                    pol, m=j, k=list(range(j)), scl=2
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_chebint_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.chebyshev.chebint(c) for c in c2d.T]).T
        res = beignet.orthax.chebyshev.chebint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.chebyshev.chebint(c) for c in c2d])
        res = beignet.orthax.chebyshev.chebint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.chebyshev.chebint(c, k=3) for c in c2d])
        res = beignet.orthax.chebyshev.chebint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestDerivative:
    def test_chebder(self):
        # check exceptions
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebder, [0], 0.5
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebder, [0], -1
        )

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.chebyshev.chebder(tgt, m=0)
            numpy.testing.assert_array_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.chebyshev.chebder(
                    beignet.orthax.chebyshev.chebint(tgt, m=j), m=j
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.chebyshev.chebder(
                    beignet.orthax.chebyshev.chebint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_chebder_axis(self):
        # check that axis keyword works
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.chebyshev.chebder(c) for c in c2d.T]).T
        res = beignet.orthax.chebyshev.chebder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.chebyshev.chebder(c) for c in c2d])
        res = beignet.orthax.chebyshev.chebder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_chebvander(self):
        # check for 1d x
        x = numpy.arange(3)
        v = beignet.orthax.chebyshev.chebvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.chebyshev.chebval(x, coef)
            )

        # check for 2d x
        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.orthax.chebyshev.chebvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.chebyshev.chebval(x, coef)
            )

    def test_chebvander2d(self):
        # also tests chebval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.orthax.chebyshev.chebvander2d(x1, x2, (1, 2))
        tgt = beignet.orthax.chebyshev.chebval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        # check shape
        van = beignet.orthax.chebyshev.chebvander2d([x1], [x2], (1, 2))
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_chebvander3d(self):
        # also tests chebval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.orthax.chebyshev.chebvander3d(x1, x2, x3, (1, 2, 3))
        tgt = beignet.orthax.chebyshev.chebval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        # check shape
        van = beignet.orthax.chebyshev.chebvander3d([x1], [x2], [x3], (1, 2, 3))
        numpy.testing.assert_(van.shape == (1, 5, 24))


class TestFitting:
    def test_chebfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebfit, [1], [1], -1
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [[1]], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [1], [[[1]]], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [1, 2], [1], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [1], [1, 2], 0
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [1], [1], 0, w=[[1]]
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [1], [1], 0, w=[1, 1]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebfit, [1], [1], (-1,)
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebfit, [1], [1], (2, -1, 6)
        )
        numpy.testing.assert_raises(
            TypeError, beignet.orthax.chebyshev.chebfit, [1], [1], ()
        )

        # Test fit
        x = numpy.linspace(0, 2)
        y = f(x)
        #
        coef3 = beignet.orthax.chebyshev.chebfit(x, y, 3)
        numpy.testing.assert_array_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef3), y
        )
        coef3 = beignet.orthax.chebyshev.chebfit(x, y, (0, 1, 2, 3))
        numpy.testing.assert_array_equal(len(coef3), 4)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef3), y
        )
        #
        coef4 = beignet.orthax.chebyshev.chebfit(x, y, 4)
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef4), y
        )
        coef4 = beignet.orthax.chebyshev.chebfit(x, y, (0, 1, 2, 3, 4))
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef4), y
        )
        # check things still work if deg is not in strict increasing
        coef4 = beignet.orthax.chebyshev.chebfit(x, y, (2, 3, 4, 1, 0))
        numpy.testing.assert_array_equal(len(coef4), 5)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef4), y
        )
        #
        coef2d = beignet.orthax.chebyshev.chebfit(x, numpy.array([y, y]).T, 3)
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        coef2d = beignet.orthax.chebyshev.chebfit(
            x, numpy.array([y, y]).T, (0, 1, 2, 3)
        )
        numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
        # test weighting
        w = numpy.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = beignet.orthax.chebyshev.chebfit(x, yw, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)
        wcoef3 = beignet.orthax.chebyshev.chebfit(x, yw, (0, 1, 2, 3), w=w)
        numpy.testing.assert_array_almost_equal(wcoef3, coef3)
        #
        wcoef2d = beignet.orthax.chebyshev.chebfit(x, numpy.array([yw, yw]).T, 3, w=w)
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        wcoef2d = beignet.orthax.chebyshev.chebfit(
            x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w
        )
        numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebfit(x, x, 1), [0, 1]
        )
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebfit(x, x, (0, 1)), [0, 1]
        )
        # test fitting only even polynomials
        x = numpy.linspace(-1, 1)
        y = f2(x)
        coef1 = beignet.orthax.chebyshev.chebfit(x, y, 4)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef1), y
        )
        coef2 = beignet.orthax.chebyshev.chebfit(x, y, (0, 2, 4))
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebval(x, coef2), y
        )
        numpy.testing.assert_array_almost_equal(coef1, coef2)


class TestInterpolate:
    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebinterpolate, self.f, -1
        )

    def test_dimensions(self):
        for deg in range(1, 5):
            numpy.testing.assert_(
                beignet.orthax.chebyshev.chebinterpolate(self.f, deg).shape
                == (deg + 1,)
            )

    def test_approximation(self):
        def powx(x, p):
            return x**p

        x = numpy.linspace(-1, 1, 10)
        for deg in range(0, 10):
            for p in range(0, deg + 1):
                c = beignet.orthax.chebyshev.chebinterpolate(powx, deg, (p,))
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.chebyshev.chebval(x, c), powx(x, p), decimal=12
                )


class TestCompanion:
    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebcompanion, []
        )
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebcompanion, [1]
        )

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            numpy.testing.assert_(
                beignet.orthax.chebyshev.chebcompanion(coef).shape == (i, i)
            )

    def test_linear_root(self):
        numpy.testing.assert_(
            beignet.orthax.chebyshev.chebcompanion([1, 2])[0, 0] == -0.5
        )


class TestGauss:
    def test_100(self):
        x, w = beignet.orthax.chebyshev.chebgauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = beignet.orthax.chebyshev.chebvander(x, 99)
        vv = numpy.dot(v.T * w, v)
        vd = 1 / numpy.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

        # check that the integral of 1 is correct
        tgt = numpy.pi
        numpy.testing.assert_array_almost_equal(w.sum(), tgt)


class TestMisc:
    def test_chebfromroots(self):
        res = beignet.orthax.chebyshev.chebfromroots([])
        numpy.testing.assert_array_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
            tgt = [0] * i + [1]
            res = beignet.orthax.chebyshev.chebfromroots(roots) * 2 ** (i - 1)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_chebroots(self):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebroots([1]), []
        )
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebroots([1, 2]), [-0.5]
        )
        for i in range(2, 5):
            tgt = numpy.linspace(-1, 1, i)
            res = beignet.orthax.chebyshev.chebroots(
                beignet.orthax.chebyshev.chebfromroots(tgt)
            )
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_chebtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebyshev.chebtrim, coef, -1
        )

        # Test results
        numpy.testing.assert_array_equal(
            beignet.orthax.chebyshev.chebtrim(coef), coef[:-1]
        )
        numpy.testing.assert_array_equal(
            beignet.orthax.chebyshev.chebtrim(coef, 1), coef[:-3]
        )
        numpy.testing.assert_array_equal(
            beignet.orthax.chebyshev.chebtrim(coef, 2), [0]
        )

    def test_chebline(self):
        numpy.testing.assert_array_equal(
            beignet.orthax.chebyshev.chebline(3, 4), [3, 4]
        )

    def test_cheb2poly(self):
        for i in range(10):
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebyshev.cheb2poly([0] * i + [1]), Tlist[i]
            )

    def test_poly2cheb(self):
        for i in range(10):
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebyshev.poly2cheb(Tlist[i]), [0] * i + [1]
            )

    def test_weight(self):
        x = numpy.linspace(-1, 1, 11)[1:-1]
        tgt = 1.0 / (numpy.sqrt(1 + x) * numpy.sqrt(1 - x))
        res = beignet.orthax.chebyshev.chebweight(x)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_chebpts1(self):
        # test exceptions
        numpy.testing.assert_raises(ValueError, beignet.orthax.chebyshev.chebpts1, 1.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.chebyshev.chebpts1, 0)

        # test points
        tgt = [0]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts1(1), tgt
        )
        tgt = [-0.70710678118654746, 0.70710678118654746]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts1(2), tgt
        )
        tgt = [-0.86602540378443871, 0, 0.86602540378443871]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts1(3), tgt
        )
        tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts1(4), tgt
        )

    def test_chebpts2(self):
        # test exceptions
        numpy.testing.assert_raises(ValueError, beignet.orthax.chebyshev.chebpts2, 1.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.chebyshev.chebpts2, 1)

        # test points
        tgt = [-1, 1]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts2(2), tgt
        )
        tgt = [-1, 0, 1]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts2(3), tgt
        )
        tgt = [-1, -0.5, 0.5, 1]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts2(4), tgt
        )
        tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebyshev.chebpts2(5), tgt
        )
