import functools

import beignet.orthax
import numpy
import numpy.testing

legcoefficients = [
    (numpy.array([1])),
    (numpy.array([0, 1])),
    (numpy.array([-1, 0, 3]) / 2),
    (numpy.array([0, -3, 0, 5]) / 2),
    (numpy.array([3, 0, -30, 0, 35]) / 8),
    (numpy.array([0, 15, 0, -70, 0, 63]) / 8),
    (numpy.array([-5, 0, 105, 0, -315, 0, 231]) / 16),
    (numpy.array([0, -35, 0, 315, 0, -693, 0, 429]) / 16),
    (numpy.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128),
    (numpy.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128),
]


def trim(x):
    return beignet.orthax.legtrim(x, tol=1e-6)


def test_legdomain():
    numpy.testing.assert_array_equal(beignet.orthax.legdomain, [-1, 1])


def test_legzero():
    numpy.testing.assert_array_equal(beignet.orthax.legzero, [0])


def test_legone():
    numpy.testing.assert_array_equal(beignet.orthax.legone, [1])


def test_legx():
    numpy.testing.assert_array_equal(beignet.orthax.legx, [0, 1])


def test_legadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.orthax.legadd([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


def test_legsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.orthax.legsub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


def test_legmulx():
    numpy.testing.assert_array_equal(trim(beignet.orthax.legmulx([0])), [0])
    numpy.testing.assert_array_equal(trim(beignet.orthax.legmulx([1])), [0, 1])
    for i in range(1, 5):
        tmp = 2 * i + 1
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
        numpy.testing.assert_array_equal(trim(beignet.orthax.legmulx(ser)), tgt)


def test_legmul():
    for i in range(5):
        pol1 = [0] * i + [1]
        x = numpy.linspace(-1, 1, 100)
        val1 = beignet.orthax.legval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.orthax.legval(x, pol2)
            pol3 = beignet.orthax.legmul(pol1, pol2)
            val3 = beignet.orthax.legval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)


def test_legdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.orthax.legadd(ci, cj)
            quo, rem = beignet.orthax.legdiv(tgt, ci)
            res = beignet.orthax.legadd(beignet.orthax.legmul(quo, ci), rem)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt), err_msg=msg)


def test_legpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.orthax.legmul, [c] * j, numpy.array([1]))
            res = beignet.orthax.legpow(c, j)
            numpy.testing.assert_array_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_legval(self):
        numpy.testing.assert_array_equal(beignet.orthax.legval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in legcoefficients]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.legval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(beignet.orthax.legval(x, [1]).shape, dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.legval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.legval(x, [1, 0, 0]).shape, dims
            )

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.legval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.legval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.leggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.leggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.leggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.leggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_legint(self):  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.legint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], -1)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], 1, [0, 0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], lbnd=[0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], scl=[0])
        numpy.testing.assert_raises(TypeError, beignet.orthax.legint, [0], axis=0.5)

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.legint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(trim(res), [0, 1])

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            legpol = beignet.orthax.poly2leg(pol)
            legint = beignet.orthax.legint(legpol, m=1, k=[i])
            res = beignet.orthax.leg2poly(legint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            legpol = beignet.orthax.poly2leg(pol)
            legint = beignet.orthax.legint(legpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legval(-1, legint), i
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            legpol = beignet.orthax.poly2leg(pol)
            legint = beignet.orthax.legint(legpol, m=1, k=[i], scl=2)
            res = beignet.orthax.leg2poly(legint)
            numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1)
                res = beignet.orthax.legint(pol, m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1, k=[k])
                res = beignet.orthax.legint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.legint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.legint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_legint_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.legint(c) for c in c2d.T]).T
        res = beignet.orthax.legint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legint(c) for c in c2d])
        res = beignet.orthax.legint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legint(c, k=3) for c in c2d])
        res = beignet.orthax.legint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        numpy.testing.assert_array_equal(beignet.orthax.legint((1, 2, 3), 0), (1, 2, 3))


class TestDerivative:
    def test_legder(self):
        numpy.testing.assert_raises(TypeError, beignet.orthax.legder, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legder, [0], -1)

        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.legder(tgt, m=0)
            numpy.testing.assert_array_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.legder(beignet.orthax.legint(tgt, m=j), m=j)
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.legder(
                    beignet.orthax.legint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.legder(c) for c in c2d.T]).T
        res = beignet.orthax.legder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legder(c) for c in c2d])
        res = beignet.orthax.legder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        numpy.testing.assert_array_equal(beignet.orthax.legder(c, 4), [0])


def test_legvander():
    x = numpy.arange(3)
    v = beignet.orthax.legvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_array_almost_equal(
            v[..., i], beignet.orthax.legval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.legvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_array_almost_equal(
            v[..., i], beignet.orthax.legval(x, coef)
        )
    numpy.testing.assert_raises(ValueError, beignet.orthax.legvander, (1, 2, 3), -1)


def test_legvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.orthax.legvander2d(x1, x2, (1, 2))
    tgt = beignet.orthax.legval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.legvander2d([x1], [x2], (1, 2))
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_legvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.orthax.legvander3d(x1, x2, x3, (1, 2, 3))
    tgt = beignet.orthax.legval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.legvander3d([x1], [x2], [x3], (1, 2, 3))
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.orthax.legfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [1], [1], 0, w=[[1]])
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [1], [1], 0, w=[1, 1])
    numpy.testing.assert_raises(ValueError, beignet.orthax.legfit, [1], [1], (-1,))
    numpy.testing.assert_raises(ValueError, beignet.orthax.legfit, [1], [1], (2, -1, 6))
    numpy.testing.assert_raises(TypeError, beignet.orthax.legfit, [1], [1], ())

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.orthax.legfit(x, y, 3)
    numpy.testing.assert_array_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef3), y)
    coef3 = beignet.orthax.legfit(x, y, (0, 1, 2, 3))
    numpy.testing.assert_array_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef3), y)

    coef4 = beignet.orthax.legfit(x, y, 4)
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef4), y)
    coef4 = beignet.orthax.legfit(x, y, (0, 1, 2, 3, 4))
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef4), y)

    coef4 = beignet.orthax.legfit(x, y, (2, 3, 4, 1, 0))
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef4), y)

    coef2d = beignet.orthax.legfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.orthax.legfit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.orthax.legfit(x, yw, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.orthax.legfit(x, yw, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.orthax.legfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.orthax.legfit(x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_array_almost_equal(beignet.orthax.legfit(x, x, 1), [0, 1])
    numpy.testing.assert_array_almost_equal(beignet.orthax.legfit(x, x, (0, 1)), [0, 1])

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.orthax.legfit(x, y, 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef1), y)
    coef2 = beignet.orthax.legfit(x, y, (0, 2, 4))
    numpy.testing.assert_array_almost_equal(beignet.orthax.legval(x, coef2), y)
    numpy.testing.assert_array_almost_equal(coef1, coef2)


def test_legcompanion():
    numpy.testing.assert_raises(ValueError, beignet.orthax.legcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.orthax.legcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.orthax.legcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.orthax.legcompanion([1, 2])[0, 0] == -0.5)


def test_leggauss():
    x, w = beignet.orthax.leggauss(100)

    v = beignet.orthax.legvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

    tgt = 2.0
    numpy.testing.assert_array_almost_equal(w.sum(), tgt)


def test_legfromroots():
    res = beignet.orthax.legfromroots([])
    numpy.testing.assert_array_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.legfromroots(roots)
        res = beignet.orthax.legval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.leg2poly(pol)[-1], 1)
        numpy.testing.assert_array_almost_equal(res, tgt)


def test_legroots():
    numpy.testing.assert_array_almost_equal(beignet.orthax.legroots([1]), [])
    numpy.testing.assert_array_almost_equal(beignet.orthax.legroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.orthax.legroots(beignet.orthax.legfromroots(tgt))
        numpy.testing.assert_array_almost_equal(trim(res), trim(tgt))


def test_legtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.legtrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.legtrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(coef, 2), [0])


def test_legline():
    numpy.testing.assert_array_equal(beignet.orthax.legline(3, 4), [3, 4])

    numpy.testing.assert_array_equal(trim(beignet.orthax.legline(3, 0)), [3])


def test_leg2poly():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.leg2poly([0] * i + [1]), legcoefficients[i]
        )


def test_poly2leg():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.poly2leg(legcoefficients[i]), [0] * i + [1]
        )


def test_legweight():
    x = numpy.linspace(-1, 1, 11)
    tgt = 1.0
    res = beignet.orthax.legweight(x)
    numpy.testing.assert_array_almost_equal(res, tgt)
