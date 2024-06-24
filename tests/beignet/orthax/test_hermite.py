import functools

import beignet.orthax
import numpy
import numpy.testing

hermcoefficients = [
    (numpy.array([1])),
    (numpy.array([0, 2])),
    (numpy.array([-2, 0, 4])),
    (numpy.array([0, -12, 0, 8])),
    (numpy.array([12, 0, -48, 0, 16])),
    (numpy.array([0, 120, 0, -160, 0, 32])),
    (numpy.array([-120, 0, 720, 0, -480, 0, 64])),
    (numpy.array([0, -1680, 0, 3360, 0, -1344, 0, 128])),
    (numpy.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])),
    (numpy.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])),
]


class TestArithmetic:
    x = numpy.linspace(-3, 3, 100)

    def test_hermadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.orthax.hermadd([0.0] * i + [1.0], [0.0] * j + [1.0])
                numpy.testing.assert_array_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                    err_msg=msg,
                )

    def test_hermsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.hermsub([0.0] * i + [1.0], [0.0] * j + [1.0])
                numpy.testing.assert_array_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                    err_msg=msg,
                )

    def test_hermmulx(self):
        x = beignet.orthax.hermmulx([0.0])
        numpy.testing.assert_array_equal(beignet.orthax.hermtrim(x, tol=1e-6), [0.0])
        numpy.testing.assert_array_equal(beignet.orthax.hermmulx([1.0]), [0.0, 0.5])
        for i in range(1, 5):
            ser = [0.0] * i + [1.0]
            tgt = [0.0] * (i - 1) + [i, 0.0, 0.5]
            numpy.testing.assert_array_equal(beignet.orthax.hermmulx(ser), tgt)

    def test_hermmul(self):
        for i in range(5):
            pol1 = [0.0] * i + [1.0]
            val1 = beignet.orthax.hermval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0.0] * j + [1.0]
                val2 = beignet.orthax.hermval(self.x, pol2)
                pol3 = beignet.orthax.hermmul(pol1, pol2)
                val3 = beignet.orthax.hermval(self.x, pol3)
                numpy.testing.assert_(
                    len(beignet.orthax.hermtrim(pol3, tol=1e-6)) == i + j + 1, msg
                )
                numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_hermdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0.0] * i + [1.0]
                cj = [0.0] * j + [1.0]
                tgt = beignet.orthax.hermadd(ci, cj)
                quo, rem = beignet.orthax.hermdiv(tgt, ci)
                res = beignet.orthax.hermadd(beignet.orthax.hermmul(quo, ci), rem)
                numpy.testing.assert_array_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                    err_msg=msg,
                )

    def test_hermpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1).astype(float)
                tgt = functools.reduce(
                    beignet.orthax.hermmul, [c] * j, numpy.array([1])
                )
                res = beignet.orthax.hermpow(c, j)
                numpy.testing.assert_array_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                    err_msg=msg,
                )


class TestEvaluation:
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_hermval(self):
        numpy.testing.assert_equal(beignet.orthax.hermval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in hermcoefficients]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.hermval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(beignet.orthax.hermval(x, [1]).shape, dims)
            numpy.testing.assert_equal(beignet.orthax.hermval(x, [1, 0]).shape, dims)
            numpy.testing.assert_equal(beignet.orthax.hermval(x, [1, 0, 0]).shape, dims)

    def test_hermval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.hermval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.hermval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.hermgrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermgrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_hermgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.hermgrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermgrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_hermint(self):  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermint, [0], -1)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermint, [0], 1, [0, 0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermint, [0], lbnd=[0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermint, [0], scl=[0])
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermint, [0], axis=0.5)

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.hermint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=1e-6), [0, 0.5]
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermpol = beignet.orthax.poly2herm(pol)
            hermint = beignet.orthax.hermint(hermpol, m=1, k=[i])
            res = beignet.orthax.herm2poly(hermint)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=1e-6),
                beignet.orthax.hermtrim(tgt, tol=1e-6),
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            hermpol = beignet.orthax.poly2herm(pol)
            hermint = beignet.orthax.hermint(hermpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermval(-1, hermint), i
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            hermpol = beignet.orthax.poly2herm(pol)
            hermint = beignet.orthax.hermint(hermpol, m=1, k=[i], scl=2)
            res = beignet.orthax.herm2poly(hermint)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=1e-6),
                beignet.orthax.hermtrim(tgt, tol=1e-6),
            )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1)
                res = beignet.orthax.hermint(pol, m=j)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1, k=[k])
                res = beignet.orthax.hermint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.hermint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.hermint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermtrim(res, tol=1e-6),
                    beignet.orthax.hermtrim(tgt, tol=1e-6),
                )

    def test_hermint_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.hermint(c) for c in c2d.T]).T
        res = beignet.orthax.hermint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.hermint(c) for c in c2d])
        res = beignet.orthax.hermint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.hermint(c, k=3) for c in c2d])
        res = beignet.orthax.hermint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)
