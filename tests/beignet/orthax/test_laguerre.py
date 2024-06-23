import functools

import beignet.orthax
import numpy
import numpy.testing

lagcoefficients = [
    (numpy.array([1]) / 1),
    (numpy.array([1, -1]) / 1),
    (numpy.array([2, -4, 1]) / 2),
    (numpy.array([6, -18, 9, -1]) / 6),
    (numpy.array([24, -96, 72, -16, 1]) / 24),
    (numpy.array([120, -600, 600, -200, 25, -1]) / 120),
    (numpy.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]


def trim(x):
    return beignet.orthax.lagtrim(x, tol=1e-6)


def test_lagdomain():
    numpy.testing.assert_array_equal(beignet.orthax.lagdomain, [0, 1])


def test_lagzero():
    numpy.testing.assert_array_equal(beignet.orthax.lagzero, [0])


def test_lagone():
    numpy.testing.assert_array_equal(beignet.orthax.lagone, [1])


def test_lagx():
    numpy.testing.assert_array_equal(beignet.orthax.lagx, [1, -1])


class TestArithmetic:
    x = numpy.linspace(-3, 3, 100)

    def test_lagadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = beignet.orthax.lagadd([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = numpy.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = beignet.orthax.lagsub([0] * i + [1], [0] * j + [1])
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagmulx(self):
        numpy.testing.assert_array_equal(trim(beignet.orthax.lagmulx([0])), [0])
        numpy.testing.assert_array_equal(trim(beignet.orthax.lagmulx([1])), [1, -1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
            numpy.testing.assert_array_almost_equal(
                trim(beignet.orthax.lagmulx(ser)), trim(tgt)
            )

    def test_lagmul(self):
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = beignet.orthax.lagval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = beignet.orthax.lagval(self.x, pol2)
                pol3 = trim(beignet.orthax.lagmul(pol1, pol2))
                val3 = beignet.orthax.lagval(self.x, pol3)
                numpy.testing.assert_(len(pol3) == i + j + 1, msg)
                numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_lagdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = beignet.orthax.lagadd(ci, cj)
                quo, rem = beignet.orthax.lagdiv(tgt, ci)
                res = beignet.orthax.lagadd(beignet.orthax.lagmul(quo, ci), rem)
                numpy.testing.assert_array_almost_equal(
                    trim(res), trim(tgt), err_msg=msg
                )

    def test_lagpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = numpy.arange(i + 1)
                tgt = functools.reduce(beignet.orthax.lagmul, [c] * j, numpy.array([1]))
                res = beignet.orthax.lagpow(c, j)
                numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_lagval(self):
        numpy.testing.assert_array_equal(beignet.orthax.lagval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [beignet.orthax.polynomial.polyval(x, c) for c in lagcoefficients]
        for i in range(7):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.lagval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(beignet.orthax.lagval(x, [1]).shape, dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.lagval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.lagval(x, [1, 0, 0]).shape, dims
            )

    def test_lagval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.lagval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.lagval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.lagval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_lagval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.lagval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.lagval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.lagval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_laggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.laggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.laggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_laggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.laggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.laggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_lagint():  # noqa:C901
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.orthax.lagint([0], m=i, k=k)
        numpy.testing.assert_array_almost_equal(trim(res), [1, -1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, m=1, k=[i])
        res = beignet.orthax.lag2poly(lagint)
        numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.lagval(-1, lagint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, m=1, k=[i], scl=2)
        res = beignet.orthax.lag2poly(lagint)
        numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1)
            res = beignet.orthax.lagint(pol, m=j)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1, k=[k])
            res = beignet.orthax.lagint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.orthax.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1, k=[k], scl=2)
            res = beignet.orthax.lagint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.lagint(c) for c in c2d.T]).T
    res = beignet.orthax.lagint(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.lagint(c) for c in c2d])
    res = beignet.orthax.lagint(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.lagint(c, k=3) for c in c2d])
    res = beignet.orthax.lagint(c2d, k=3, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_lagder():
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.orthax.lagder(tgt, m=0)
        numpy.testing.assert_array_equal(trim(res), trim(tgt))

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.lagder(beignet.orthax.lagint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.lagder(
                beignet.orthax.lagint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.lagder(c) for c in c2d.T]).T
    res = beignet.orthax.lagder(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.lagder(c) for c in c2d])
    res = beignet.orthax.lagder(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_lagvander():
    x = numpy.arange(3)
    v = beignet.orthax.lagvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_array_almost_equal(
            v[..., i], beignet.orthax.lagval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.lagvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_array_almost_equal(
            v[..., i], beignet.orthax.lagval(x, coef)
        )


def test_lagvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.orthax.lagvander2d(x1, x2, (1, 2))
    tgt = beignet.orthax.lagval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.lagvander2d([x1], [x2], (1, 2))
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_lagvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.orthax.lagvander3d(x1, x2, x3, (1, 2, 3))
    tgt = beignet.orthax.lagval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.lagvander3d([x1], [x2], [x3], (1, 2, 3))
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(ValueError, beignet.orthax.lagfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [1], [1], 0, w=[[1]])
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [1], [1], 0, w=[1, 1])
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagfit, [1], [1], (-1,))
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagfit, [1], [1], (2, -1, 6))
    numpy.testing.assert_raises(TypeError, beignet.orthax.lagfit, [1], [1], ())

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.orthax.lagfit(x, y, 3)
    numpy.testing.assert_array_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagval(x, coef3), y)
    coef3 = beignet.orthax.lagfit(x, y, (0, 1, 2, 3))
    numpy.testing.assert_array_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagval(x, coef3), y)

    coef4 = beignet.orthax.lagfit(x, y, 4)
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagval(x, coef4), y)
    coef4 = beignet.orthax.lagfit(x, y, (0, 1, 2, 3, 4))
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagval(x, coef4), y)

    coef2d = beignet.orthax.lagfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.orthax.lagfit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.orthax.lagfit(x, yw, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.orthax.lagfit(x, yw, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.orthax.lagfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.orthax.lagfit(x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagfit(x, x, 1), [1, -1])
    numpy.testing.assert_array_almost_equal(
        beignet.orthax.lagfit(x, x, (0, 1)), [1, -1]
    )


def test_lagcompanion(self):
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.orthax.lagcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.orthax.lagcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.orthax.lagcompanion([1, 2])[0, 0] == 1.5)


def test_laggauss():
    x, w = beignet.orthax.laggauss(100)

    v = beignet.orthax.lagvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

    tgt = 1.0
    numpy.testing.assert_array_almost_equal(w.sum(), tgt)


def test_lagfromroots():
    res = beignet.orthax.lagfromroots([])
    numpy.testing.assert_array_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.lagfromroots(roots)
        res = beignet.orthax.lagval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.lag2poly(pol)[-1], 1)
        numpy.testing.assert_array_almost_equal(res, tgt)


def test_lagroots():
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagroots([1]), [])
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagroots([0, 1]), [1])
    for i in range(2, 5):
        tgt = numpy.linspace(0, 3, i)
        res = beignet.orthax.lagroots(beignet.orthax.lagfromroots(tgt))
        numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))


def test_lagtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.lagtrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.lagtrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.lagtrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.lagtrim(coef, 2), [0])


def test_lagline():
    numpy.testing.assert_array_equal(beignet.orthax.lagline(3, 4), [7, -4])


def test_lag2poly():
    for i in range(7):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.lag2poly([0] * i + [1]), lagcoefficients[i]
        )


def test_poly2lag():
    for i in range(7):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.poly2lag(lagcoefficients[i]), [0] * i + [1]
        )


def test_lagweight():
    x = numpy.linspace(0, 10, 11)
    tgt = numpy.exp(-x)
    res = beignet.orthax.lagweight(x)
    numpy.testing.assert_array_almost_equal(res, tgt)
