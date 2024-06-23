import functools

import beignet.orthax
import numpy
import numpy.testing

H0 = numpy.array([1])
H1 = numpy.array([0, 2])
H2 = numpy.array([-2, 0, 4])
H3 = numpy.array([0, -12, 0, 8])
H4 = numpy.array([12, 0, -48, 0, 16])
H5 = numpy.array([0, 120, 0, -160, 0, 32])
H6 = numpy.array([-120, 0, 720, 0, -480, 0, 64])
H7 = numpy.array([0, -1680, 0, 3360, 0, -1344, 0, 128])
H8 = numpy.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
H9 = numpy.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])

Hlist = [H0, H1, H2, H3, H4, H5, H6, H7, H8, H9]


def trim(x):
    return beignet.orthax.hermtrim(x, tol=1e-6)


def test_hermdomain():
    numpy.testing.assert_array_equal(beignet.orthax.hermdomain, numpy.array([-1, 1]))


def test_hermzero():
    numpy.testing.assert_array_equal(beignet.orthax.hermzero, numpy.array([0]))


def test_hermone():
    numpy.testing.assert_array_equal(beignet.orthax.hermone, numpy.array([1]))


def test_hermx():
    numpy.testing.assert_array_equal(beignet.orthax.hermx, numpy.array([0, 0.5]))


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
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.hermsub([0.0] * i + [1.0], [0.0] * j + [1.0])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermmulx(self):
        numpy.testing.assert_array_equal(trim(beignet.orthax.hermmulx([0.0])), [0.0])
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
                numpy.testing.assert_(len(trim(pol3)) == i + j + 1, msg)
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
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1).astype(float)
                tgt = functools.reduce(
                    beignet.orthax.hermmul, [c] * j, numpy.array([1])
                )
                res = beignet.orthax.hermpow(c, j)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_hermval(self):
        numpy.testing.assert_equal(beignet.orthax.hermval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in Hlist]
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
            numpy.testing.assert_array_almost_equal(trim(res), [0, 0.5])

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermpol = beignet.orthax.poly2herm(pol)
            hermint = beignet.orthax.hermint(hermpol, m=1, k=[i])
            res = beignet.orthax.herm2poly(hermint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

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
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1)
                res = beignet.orthax.hermint(pol, m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1, k=[k])
                res = beignet.orthax.hermint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.hermint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.hermint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

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


class TestDerivative:
    def test_hermder(self):
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermder, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermder, [0], -1)

        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.hermder(tgt, m=0)
            numpy.testing.assert_array_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.hermder(beignet.orthax.hermint(tgt, m=j), m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.hermder(
                    beignet.orthax.hermint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_hermder_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.hermder(c) for c in c2d.T]).T
        res = beignet.orthax.hermder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.hermder(c) for c in c2d])
        res = beignet.orthax.hermder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)


class TestVander:
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_hermvander(self):
        x = numpy.arange(3)
        v = beignet.orthax.hermvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.hermval(x, coef)
            )

        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.orthax.hermvander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.hermval(x, coef)
            )

    def test_hermvander2d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.orthax.hermvander2d(x1, x2, (1, 2))
        tgt = beignet.orthax.hermval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.hermvander2d([x1], [x2], (1, 2))
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_hermvander3d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.orthax.hermvander3d(x1, x2, x3, (1, 2, 3))
        tgt = beignet.orthax.hermval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.hermvander3d([x1], [x2], [x3], (1, 2, 3))
        numpy.testing.assert_(van.shape == (1, 5, 24))


def test_hermfit(self):
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.orthax.hermfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [1], [1], 0, w=[[1]])
    numpy.testing.assert_raises(
        TypeError, beignet.orthax.hermfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermfit, [1], [1], (-1))
    numpy.testing.assert_raises(
        ValueError, beignet.orthax.hermfit, [1], [1], (2, -1, 6)
    )
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermfit, [1], [1], ())

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.orthax.hermfit(x, y, 3)
    numpy.testing.assert_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef3), y)
    coef3 = beignet.orthax.hermfit(x, y, (0, 1, 2, 3))
    numpy.testing.assert_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef3), y)

    coef4 = beignet.orthax.hermfit(x, y, 4)
    numpy.testing.assert_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef4), y)
    coef4 = beignet.orthax.hermfit(x, y, (0, 1, 2, 3, 4))
    numpy.testing.assert_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef4), y)

    coef4 = beignet.orthax.hermfit(x, y, (2, 3, 4, 1, 0))
    numpy.testing.assert_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef4), y)

    coef2d = beignet.orthax.hermfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.orthax.hermfit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.orthax.hermfit(x, yw, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.orthax.hermfit(x, yw, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.orthax.hermfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.orthax.hermfit(x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermfit(x, x, 1), [0, 0.5])
    numpy.testing.assert_array_almost_equal(
        beignet.orthax.hermfit(x, x, (0, 1)), [0, 0.5]
    )

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.orthax.hermfit(x, y, 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef1), y)
    coef2 = beignet.orthax.hermfit(x, y, (0, 2, 4))
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(x, coef2), y)
    numpy.testing.assert_array_almost_equal(coef1, coef2)


def test_hermcompanion(self):
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.orthax.hermcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.orthax.hermcompanion([1, 2])[0, 0] == -0.25)


def test_hermgauss():
    x, w = beignet.orthax.hermgauss(100)

    v = beignet.orthax.hermvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

    tgt = numpy.sqrt(numpy.pi)
    numpy.testing.assert_array_almost_equal(w.sum(), tgt)


def test_hermfromroots():
    res = beignet.orthax.hermfromroots([])
    numpy.testing.assert_array_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.hermfromroots(roots)
        res = beignet.orthax.hermval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.herm2poly(pol)[-1], 1)
        numpy.testing.assert_array_almost_equal(res, tgt)


def test_hermroots():
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermroots([1]), [])
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermroots([1, 1]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.orthax.hermroots(beignet.orthax.hermfromroots(tgt))
        numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))


def test_hermtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.hermtrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.hermtrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.hermtrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.hermtrim(coef, 2), [0])


def test_hermline():
    numpy.testing.assert_array_equal(beignet.orthax.hermline(3, 4), [3, 2])


def test_herm2poly():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.herm2poly([0] * i + [1]), Hlist[i]
        )


def test_poly2herm():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            trim(beignet.orthax.poly2herm(Hlist[i])), [0] * i + [1]
        )


def test_hermweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-(x**2))
    res = beignet.orthax.hermweight(x)
    numpy.testing.assert_array_almost_equal(res, tgt)
