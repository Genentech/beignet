import functools

import beignet.orthax
import numpy
import numpy.testing

chebcoefficients = [
    [1],
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8],
    [0, 5, 0, -20, 0, 16],
    [-1, 0, 18, 0, -48, 0, 32],
    [0, -7, 0, 56, 0, -112, 0, 64],
    [1, 0, -32, 0, 160, 0, -256, 0, 128],
    [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
]

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

hermecoefficients = [
    (numpy.array([1])),
    (numpy.array([0, 1])),
    (numpy.array([-1, 0, 1])),
    (numpy.array([0, -3, 0, 1])),
    (numpy.array([3, 0, -6, 0, 1])),
    (numpy.array([0, 15, 0, -10, 0, 1])),
    (numpy.array([-15, 0, 45, 0, -15, 0, 1])),
    (numpy.array([0, -105, 0, 105, 0, -21, 0, 1])),
    (numpy.array([105, 0, -420, 0, 210, 0, -28, 0, 1])),
    (numpy.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])),
]

lagcoefficients = [
    (numpy.array([1]) / 1),
    (numpy.array([1, -1]) / 1),
    (numpy.array([2, -4, 1]) / 2),
    (numpy.array([6, -18, 9, -1]) / 6),
    (numpy.array([24, -96, 72, -16, 1]) / 24),
    (numpy.array([120, -600, 600, -200, 25, -1]) / 120),
    (numpy.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]

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

polycoefficients = [
    [1],
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8],
    [0, 5, 0, -20, 0, 16],
    [-1, 0, 18, 0, -48, 0, 32],
    [0, -7, 0, 56, 0, -112, 0, 64],
    [1, 0, -32, 0, 160, 0, -256, 0, 128],
    [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
]


def test__cseries_to_zseries():
    for i in range(5):
        inp = numpy.array([2] + [1] * i, numpy.double)
        tgt = numpy.array([0.5] * i + [2] + [0.5] * i, numpy.double)
        res = beignet.orthax._cseries_to_zseries(inp)
        numpy.testing.assert_array_equal(res, tgt)


def test__zseries_to_cseries():
    for i in range(5):
        inp = numpy.array([0.5] * i + [2] + [0.5] * i, numpy.double)
        tgt = numpy.array([2] + [1] * i, numpy.double)
        res = beignet.orthax._zseries_to_cseries(inp)
        numpy.testing.assert_array_equal(res, tgt)


def test_chebdomain():
    numpy.testing.assert_array_equal(beignet.orthax.chebdomain, [-1, 1])


def test_chebzero():
    numpy.testing.assert_array_equal(beignet.orthax.chebzero, [0])


def test_chebone():
    numpy.testing.assert_array_equal(beignet.orthax.chebone, [1])


def test_chebx():
    numpy.testing.assert_array_equal(beignet.orthax.chebx, [0, 1])


def test_chebadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.orthax.chebadd([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_chebsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.orthax.chebsub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_chebmulx():
    x = beignet.orthax.chebmulx([0])
    numpy.testing.assert_array_equal(beignet.orthax.chebtrim(x, tol=1e-6), [0])
    x1 = beignet.orthax.chebmulx([1])
    numpy.testing.assert_array_equal(beignet.orthax.chebtrim(x1, tol=1e-6), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [0.5, 0, 0.5]
        x2 = beignet.orthax.chebmulx(ser)
        numpy.testing.assert_array_equal(beignet.orthax.chebtrim(x2, tol=1e-6), tgt)


def test_chebmul():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(i + j + 1)
            tgt[i + j] += 0.5
            tgt[abs(i - j)] += 0.5
            res = beignet.orthax.chebmul([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.orthax.chebadd(ci, cj)
            quo, rem = beignet.orthax.chebdiv(tgt, ci)
            res = beignet.orthax.chebadd(beignet.orthax.chebmul(quo, ci), rem)
            numpy.testing.assert_array_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_chebpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.orthax.chebmul, [c] * j, numpy.array([1]))
            res = beignet.orthax.chebpow(c, j)
            numpy.testing.assert_array_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_chebint():
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.orthax.chebint([0], m=i, k=k)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebtrim(res, tol=1e-6), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        chebpol = beignet.orthax.poly2cheb(pol)
        chebint = beignet.orthax.chebint(chebpol, m=1, k=[i])
        res = beignet.orthax.cheb2poly(chebint)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebtrim(res, tol=1e-6),
            beignet.orthax.chebtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        chebpol = beignet.orthax.poly2cheb(pol)
        chebint = beignet.orthax.chebint(chebpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(-1, chebint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        chebpol = beignet.orthax.poly2cheb(pol)
        chebint = beignet.orthax.chebint(chebpol, m=1, k=[i], scl=2)
        res = beignet.orthax.cheb2poly(chebint)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebtrim(res, tol=1e-6),
            beignet.orthax.chebtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.orthax.chebint(tgt, m=1)
            res = beignet.orthax.chebint(pol, m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.chebint(tgt, m=1, k=[k])
            res = beignet.orthax.chebint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.chebint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.orthax.chebint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.chebint(tgt, m=1, k=[k], scl=2)
            res = beignet.orthax.chebint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.chebint(c) for c in c2d.T]).T
    res = beignet.orthax.chebint(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.chebint(c) for c in c2d])
    res = beignet.orthax.chebint(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.chebint(c, k=3) for c in c2d])
    res = beignet.orthax.chebint(c2d, k=3, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_chebder():
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.orthax.chebder(tgt, m=0)
        numpy.testing.assert_array_equal(
            beignet.orthax.chebtrim(res, tol=1e-6),
            beignet.orthax.chebtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.chebder(beignet.orthax.chebint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.chebder(
                beignet.orthax.chebint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebtrim(res, tol=1e-6),
                beignet.orthax.chebtrim(tgt, tol=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.chebder(c) for c in c2d.T]).T
    res = beignet.orthax.chebder(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.chebder(c) for c in c2d])
    res = beignet.orthax.chebder(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_chebvander():
    x = numpy.arange(3)
    v = beignet.orthax.chebvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_array_almost_equal(
            v[..., i], beignet.orthax.chebval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.chebvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_array_almost_equal(
            v[..., i], beignet.orthax.chebval(x, coef)
        )


def test_chebvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.orthax.chebvander2d(x1, x2, (1, 2))
    tgt = beignet.orthax.chebval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.chebvander2d([x1], [x2], (1, 2))
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_chebvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.orthax.chebvander3d(x1, x2, x3, (1, 2, 3))
    tgt = beignet.orthax.chebval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.chebvander3d([x1], [x2], [x3], (1, 2, 3))
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_chebfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.orthax.chebfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [1], [1], 0, w=[[1]])
    numpy.testing.assert_raises(
        TypeError, beignet.orthax.chebfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebfit, [1], [1], (-1,))
    numpy.testing.assert_raises(
        ValueError, beignet.orthax.chebfit, [1], [1], (2, -1, 6)
    )
    numpy.testing.assert_raises(TypeError, beignet.orthax.chebfit, [1], [1], ())

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.orthax.chebfit(x, y, 3)
    numpy.testing.assert_array_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef3), y)
    coef3 = beignet.orthax.chebfit(x, y, (0, 1, 2, 3))
    numpy.testing.assert_array_equal(len(coef3), 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef3), y)

    coef4 = beignet.orthax.chebfit(x, y, 4)
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef4), y)
    coef4 = beignet.orthax.chebfit(x, y, (0, 1, 2, 3, 4))
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef4), y)

    coef4 = beignet.orthax.chebfit(x, y, (2, 3, 4, 1, 0))
    numpy.testing.assert_array_equal(len(coef4), 5)
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef4), y)

    coef2d = beignet.orthax.chebfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.orthax.chebfit(x, numpy.array([y, y]).T, (0, 1, 2, 3))
    numpy.testing.assert_array_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.orthax.chebfit(x, yw, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.orthax.chebfit(x, yw, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.orthax.chebfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.orthax.chebfit(x, numpy.array([yw, yw]).T, (0, 1, 2, 3), w=w)
    numpy.testing.assert_array_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebfit(x, x, 1), [0, 1])
    numpy.testing.assert_array_almost_equal(
        beignet.orthax.chebfit(x, x, (0, 1)), [0, 1]
    )

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.orthax.chebfit(x, y, 4)
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef1), y)
    coef2 = beignet.orthax.chebfit(x, y, (0, 2, 4))
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebval(x, coef2), y)
    numpy.testing.assert_array_almost_equal(coef1, coef2)


def test_chebcompanion():
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.orthax.chebcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.orthax.chebcompanion([1, 2])[0, 0] == -0.5)


def test_chebgauss():
    x, w = beignet.orthax.chebgauss(100)

    v = beignet.orthax.chebvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

    tgt = numpy.pi
    numpy.testing.assert_array_almost_equal(w.sum(), tgt)


def test_chebfromroots():
    res = beignet.orthax.chebfromroots([])
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebtrim(res, tol=1e-6), [1])
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        tgt = [0] * i + [1]
        res = beignet.orthax.chebfromroots(roots) * 2 ** (i - 1)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebtrim(res, tol=1e-6),
            beignet.orthax.chebtrim(tgt, tol=1e-6),
        )


def test_chebroots():
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebroots([1]), [])
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.orthax.chebroots(beignet.orthax.chebfromroots(tgt))
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.chebtrim(res, tol=1e-6),
            beignet.orthax.chebtrim(tgt, tol=1e-6),
        )


def test_chebtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.chebtrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.chebtrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.chebtrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.chebtrim(coef, 2), [0])


def test_chebline():
    numpy.testing.assert_array_equal(beignet.orthax.chebline(3, 4), [3, 4])


def test_cheb2poly():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.cheb2poly([0] * i + [1]), chebcoefficients[i]
        )


def test_poly2cheb():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.poly2cheb(chebcoefficients[i]), [0] * i + [1]
        )


def test_chebweight():
    x = numpy.linspace(-1, 1, 11)[1:-1]
    tgt = 1.0 / (numpy.sqrt(1 + x) * numpy.sqrt(1 - x))
    res = beignet.orthax.chebweight(x)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_chebpts1():
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebpts1, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebpts1, 0)

    tgt = [0]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts1(1), tgt)
    tgt = [-0.70710678118654746, 0.70710678118654746]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts1(2), tgt)
    tgt = [-0.86602540378443871, 0, 0.86602540378443871]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts1(3), tgt)
    tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts1(4), tgt)


def test_chebpts2():
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebpts2, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.chebpts2, 1)

    tgt = [-1, 1]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts2(2), tgt)
    tgt = [-1, 0, 1]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts2(3), tgt)
    tgt = [-1, -0.5, 0.5, 1]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts2(4), tgt)
    tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
    numpy.testing.assert_array_almost_equal(beignet.orthax.chebpts2(5), tgt)


def test_hermdomain():
    numpy.testing.assert_array_equal(beignet.orthax.hermdomain, numpy.array([-1, 1]))


def test_hermzero():
    numpy.testing.assert_array_equal(beignet.orthax.hermzero, numpy.array([0]))


def test_hermone():
    numpy.testing.assert_array_equal(beignet.orthax.hermone, numpy.array([1]))


def test_hermx():
    numpy.testing.assert_array_equal(beignet.orthax.hermx, numpy.array([0, 0.5]))


def test_hermder(self):
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.orthax.hermder(tgt, m=0)
        numpy.testing.assert_array_equal(
            beignet.orthax.hermtrim(res, tol=1e-6),
            beignet.orthax.hermtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.hermder(beignet.orthax.hermint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=1e-6),
                beignet.orthax.hermtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.hermder(
                beignet.orthax.hermint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=1e-6),
                beignet.orthax.hermtrim(tgt, tol=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.hermder(c) for c in c2d.T]).T
    res = beignet.orthax.hermder(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.hermder(c) for c in c2d])
    res = beignet.orthax.hermder(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_hermvander():
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


def test_hermvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.orthax.hermvander2d(x1, x2, (1, 2))
    tgt = beignet.orthax.hermval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.hermvander2d([x1], [x2], (1, 2))
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_hermvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.orthax.hermvander3d(x1, x2, x3, (1, 2, 3))
    tgt = beignet.orthax.hermval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.hermvander3d([x1], [x2], [x3], (1, 2, 3))
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_hermfit():
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


def test_hermcompanion():
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
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermtrim(res, tol=1e-6), [1])
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
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.hermtrim(res, tol=1e-6),
            beignet.orthax.hermtrim(tgt, tol=1e-6),
        )


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
            beignet.orthax.herm2poly([0] * i + [1]), hermcoefficients[i]
        )


def test_poly2herm():
    for i in range(10):
        x = beignet.orthax.poly2herm(hermcoefficients[i])
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.hermtrim(x, tol=1e-6), [0] * i + [1]
        )


def test_hermweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-(x**2))
    res = beignet.orthax.hermweight(x)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_hermedomain():
    numpy.testing.assert_array_equal(beignet.orthax.hermedomain, [-1, 1])


def test_hermezero():
    numpy.testing.assert_array_equal(beignet.orthax.hermezero, [0])


def test_hermeone():
    numpy.testing.assert_array_equal(beignet.orthax.hermeone, [1])


def test_hermex():
    numpy.testing.assert_array_equal(beignet.orthax.hermex, [0, 1])


def test_hermefromroots():
    res = beignet.orthax.hermefromroots([])
    numpy.testing.assert_array_almost_equal(
        beignet.orthax.hermetrim(res, tol=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.hermefromroots(roots)
        res = beignet.orthax.hermeval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.herme2poly(pol)[-1], 1)
        numpy.testing.assert_array_almost_equal(res, tgt)


def test_hermeroots():
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermeroots([1]), [])
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermeroots([1, 1]), [-1])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.orthax.hermeroots(beignet.orthax.hermefromroots(tgt))
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.hermetrim(res, tol=1e-6),
            beignet.orthax.hermetrim(tgt, tol=1e-6),
        )


def test_hermetrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.hermetrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.hermetrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.hermetrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.hermetrim(coef, 2), [0])


def test_hermeline():
    numpy.testing.assert_array_equal(beignet.orthax.hermeline(3, 4), [3, 4])


def test_herme2poly():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.herme2poly([0] * i + [1]), hermecoefficients[i]
        )


def test_poly2herme():
    for i in range(10):
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.poly2herme(hermecoefficients[i]), [0] * i + [1]
        )


def test_hermeweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-0.5 * x**2)
    res = beignet.orthax.hermeweight(x)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_lagdomain():
    numpy.testing.assert_array_equal(beignet.orthax.lagdomain, [0, 1])


def test_lagzero():
    numpy.testing.assert_array_equal(beignet.orthax.lagzero, [0])


def test_lagone():
    numpy.testing.assert_array_equal(beignet.orthax.lagone, [1])


def test_lagx():
    numpy.testing.assert_array_equal(beignet.orthax.lagx, [1, -1])


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
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.lagtrim(res, tol=1e-6), [1, -1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, m=1, k=[i])
        res = beignet.orthax.lag2poly(lagint)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.lagtrim(res, tol=1e-6), beignet.orthax.lagtrim(tgt, tol=1e-6)
        )

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
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.lagtrim(res, tol=1e-6), beignet.orthax.lagtrim(tgt, tol=1e-6)
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1)
            res = beignet.orthax.lagint(pol, m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1, k=[k])
            res = beignet.orthax.lagint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.orthax.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.lagint(tgt, m=1, k=[k], scl=2)
            res = beignet.orthax.lagint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
            )

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
        numpy.testing.assert_array_equal(
            beignet.orthax.lagtrim(res, tol=1e-6), beignet.orthax.lagtrim(tgt, tol=1e-6)
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.lagder(beignet.orthax.lagint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.lagder(
                beignet.orthax.lagint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
            )

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
    numpy.testing.assert_array_almost_equal(beignet.orthax.lagtrim(res, tol=1e-6), [1])
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
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.lagtrim(res, tol=1e-6), beignet.orthax.lagtrim(tgt, tol=1e-6)
        )


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
            numpy.testing.assert_array_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_legsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.orthax.legsub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_legmulx():
    x = beignet.orthax.legmulx([0])
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(x, tol=1e-6), [0])
    x1 = beignet.orthax.legmulx([1])
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(x1, tol=1e-6), [0, 1])
    for i in range(1, 5):
        tmp = 2 * i + 1
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
        x2 = beignet.orthax.legmulx(ser)
        numpy.testing.assert_array_equal(beignet.orthax.legtrim(x2, tol=1e-6), tgt)


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
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_legpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.orthax.legmul, [c] * j, numpy.array([1]))
            res = beignet.orthax.legpow(c, j)
            numpy.testing.assert_array_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


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
    numpy.testing.assert_array_almost_equal(beignet.orthax.legtrim(res, tol=1e-6), [1])
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
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legtrim(res, tol=1e-6), beignet.orthax.legtrim(tgt, tol=1e-6)
        )


def test_legtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.legtrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.legtrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(coef, 2), [0])


def test_legline():
    numpy.testing.assert_array_equal(beignet.orthax.legline(3, 4), [3, 4])

    x = beignet.orthax.legline(3, 0)
    numpy.testing.assert_array_equal(beignet.orthax.legtrim(x, tol=1e-6), [3])


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


def test_polydomain():
    numpy.testing.assert_equal(beignet.orthax.polydomain, numpy.array([-1, 1]))


def test_polyzero():
    numpy.testing.assert_equal(beignet.orthax.polyzero, numpy.array([0]))


def test_polyone():
    numpy.testing.assert_equal(beignet.orthax.polyone, numpy.array([1]))


def test_polyx():
    numpy.testing.assert_equal(beignet.orthax.polyx, numpy.array([0, 1]))


def test_polyadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.orthax.polyadd([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_polysub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.orthax.polysub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_polymulx():
    numpy.testing.assert_array_equal(beignet.orthax.polymulx([0]), [0, 0])
    numpy.testing.assert_array_equal(beignet.orthax.polymulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i + 1) + [1]
        numpy.testing.assert_array_equal(beignet.orthax.polymulx(ser), tgt)


def test_polymul():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(i + j + 1)
            tgt[i + j] += 1
            res = beignet.orthax.polymul([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_polydiv():
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
            numpy.testing.assert_array_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_polypow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.orthax.polymul, [c] * j, numpy.array([1]))
            res = beignet.orthax.polypow(c, j)
            numpy.testing.assert_array_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_polyint():
    numpy.testing.assert_raises(TypeError, beignet.orthax.polyint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.polyint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.orthax.polyint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.orthax.polyint([0], m=i, k=k)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polytrim(res, tol=1e-6), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        res = beignet.orthax.polyint(pol, m=1, k=[i])
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polytrim(res, tol=1e-6),
            beignet.orthax.polytrim(tgt, tol=1e-6),
        )

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
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polytrim(res, tol=1e-6),
            beignet.orthax.polytrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.orthax.polyint(tgt, m=1)
            res = beignet.orthax.polyint(pol, m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.polyint(tgt, m=1, k=[k])
            res = beignet.orthax.polyint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.polyint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.orthax.polyint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.polyint(tgt, m=1, k=[k], scl=2)
            res = beignet.orthax.polyint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
            )

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


def test_polyder():
    numpy.testing.assert_raises(TypeError, beignet.orthax.polyder, [0], 0.5)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.orthax.polyder(tgt, m=0)
        numpy.testing.assert_array_equal(
            beignet.orthax.polytrim(res, tol=1e-6),
            beignet.orthax.polytrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.polyder(beignet.orthax.polyint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.polyder(
                beignet.orthax.polyint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.polytrim(res, tol=1e-6),
                beignet.orthax.polytrim(tgt, tol=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.polyder(c) for c in c2d.T]).T
    res = beignet.orthax.polyder(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.polyder(c) for c in c2d])
    res = beignet.orthax.polyder(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_polyvander():
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

    x = numpy.arange(3)
    numpy.testing.assert_raises(ValueError, beignet.orthax.polyvander, x, -1)


def test_polyvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.orthax.polyvander2d(x1, x2, (1, 2))
    tgt = beignet.orthax.polyval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.polyvander2d([x1], [x2], (1, 2))
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_polyvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.orthax.polyvander3d(x1, x2, x3, (1, 2, 3))
    tgt = beignet.orthax.polyval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.polyvander3d([x1], [x2], [x3], (1, 2, 3))
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_polycompanion():
    numpy.testing.assert_raises(ValueError, beignet.orthax.polycompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.orthax.polycompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.orthax.polycompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.orthax.polycompanion([1, 2])[0, 0] == -0.5)


def test_polyfromroots():
    res = beignet.orthax.polyfromroots([])
    numpy.testing.assert_array_almost_equal(beignet.orthax.polytrim(res, tol=1e-6), [1])
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        tgt = polycoefficients[i]
        res = beignet.orthax.polyfromroots(roots) * 2 ** (i - 1)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polytrim(res, tol=1e-6),
            beignet.orthax.polytrim(tgt, tol=1e-6),
        )


def test_polyroots():
    numpy.testing.assert_array_almost_equal(beignet.orthax.polyroots([1]), [])
    numpy.testing.assert_array_almost_equal(beignet.orthax.polyroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.orthax.polyroots(beignet.orthax.polyfromroots(tgt))
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polytrim(res, tol=1e-6),
            beignet.orthax.polytrim(tgt, tol=1e-6),
        )


def test_polyfit():
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
    numpy.testing.assert_raises(TypeError, beignet.orthax.polyfit, [1], [1], 0, w=[[1]])
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


def test_polytrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.orthax.polytrim, coef, -1)

    numpy.testing.assert_array_equal(beignet.orthax.polytrim(coef), coef[:-1])
    numpy.testing.assert_array_equal(beignet.orthax.polytrim(coef, 1), coef[:-3])
    numpy.testing.assert_array_equal(beignet.orthax.polytrim(coef, 2), [0])


def test_polyline():
    numpy.testing.assert_array_equal(beignet.orthax.polyline(3, 4), [3, 4])


def test_polyline_zero():
    numpy.testing.assert_array_equal(beignet.orthax.polyline(3, 0), [3, 0])


def test_trimseq():
    for _ in range(5):
        numpy.testing.assert_equal(beignet.orthax.trimseq([1] + [0] * 5), [1])


def test_trimcoef():
    coef = numpy.array([2, -1, 1, 0])

    numpy.testing.assert_raises(ValueError, beignet.orthax.trimcoef, coef, -1)

    numpy.testing.assert_equal(beignet.orthax.trimcoef(coef), coef[:-1])
    numpy.testing.assert_equal(beignet.orthax.trimcoef(coef, 1), coef[:-3])
    numpy.testing.assert_equal(beignet.orthax.trimcoef(coef, 2), numpy.array([0]))


def test_vander_nd_exception():
    numpy.testing.assert_raises(
        ValueError, beignet.orthax._vander_nd, (), (1, 2, 3), [90]
    )

    numpy.testing.assert_raises(ValueError, beignet.orthax._vander_nd, (), (), [90.65])

    numpy.testing.assert_raises(ValueError, beignet.orthax._vander_nd, (), (), [])


def test_pow_too_large():
    numpy.testing.assert_raises(ValueError, beignet.orthax._pow, (), [1, 2, 3], 5, 4)


def test_getdomain():
    x = [1, 10, 3, -1]
    tgt = [-1, 10]
    res = beignet.orthax.getdomain(x)
    numpy.testing.assert_array_equal(res, tgt)

    x = [1 + 1j, 1 - 1j, 0, 2]
    tgt = [-1j, 2 + 1j]
    res = beignet.orthax.getdomain(x)
    numpy.testing.assert_array_equal(res, tgt)


def test_mapdomain():
    dom1 = [0, 4]
    dom2 = [1, 3]
    tgt = dom2
    res = beignet.orthax.mapdomain(dom1, dom1, dom2)
    numpy.testing.assert_array_equal(res, tgt)

    dom1 = [0 - 1j, 2 + 1j]
    dom2 = [-2, 2]
    tgt = dom2
    x = dom1
    res = beignet.orthax.mapdomain(x, dom1, dom2)
    numpy.testing.assert_array_equal(res, tgt)

    dom1 = [0, 4]
    dom2 = [1, 3]
    tgt = numpy.array([dom2, dom2])
    x = numpy.array([dom1, dom1])
    res = beignet.orthax.mapdomain(x, dom1, dom2)
    numpy.testing.assert_array_equal(res, tgt)


def test_mapparms():
    numpy.testing.assert_array_equal(beignet.orthax.mapparms([0, 4], [1, 3]), [1, 0.5])

    numpy.testing.assert_array_equal(
        beignet.orthax.mapparms([0 - 1j, 2 + 1j], [-2, 2]),
        [-1 + 1j, 1 - 1j],
    )


def test_hermeadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.orthax.hermeadd([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_hermesub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.orthax.hermesub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_hermemulx():
    x = beignet.orthax.hermemulx([0])
    numpy.testing.assert_array_equal(beignet.orthax.hermetrim(x, tol=1e-6), [0])
    x1 = beignet.orthax.hermemulx([1])
    numpy.testing.assert_array_equal(beignet.orthax.hermetrim(x1, tol=1e-6), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i, 0, 1]
        x2 = beignet.orthax.hermemulx(ser)
        numpy.testing.assert_array_equal(beignet.orthax.hermetrim(x2, tol=1e-6), tgt)


def test_hermemul():
    x = numpy.linspace(-3, 3, 100)
    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = beignet.orthax.hermeval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.orthax.hermeval(x, pol2)
            pol3 = beignet.orthax.hermemul(pol1, pol2)
            val3 = beignet.orthax.hermeval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)


def test_hermediv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.orthax.hermeadd(ci, cj)
            quo, rem = beignet.orthax.hermediv(tgt, ci)
            res = beignet.orthax.hermeadd(beignet.orthax.hermemul(quo, ci), rem)
            numpy.testing.assert_array_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_hermepow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.orthax.hermemul, [c] * j, numpy.array([1]))
            res = beignet.orthax.hermepow(c, j)
            numpy.testing.assert_array_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_lagadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.orthax.lagadd([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.orthax.lagsub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_array_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_lagmulx():
    x = beignet.orthax.lagmulx([0])
    numpy.testing.assert_array_equal(beignet.orthax.lagtrim(x, tol=1e-6), [0])
    x1 = beignet.orthax.lagmulx([1])
    numpy.testing.assert_array_equal(beignet.orthax.lagtrim(x1, tol=1e-6), [1, -1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
        x2 = beignet.orthax.lagmulx(ser)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.lagtrim(x2, tol=1e-6),
            beignet.orthax.lagtrim(tgt, tol=1e-6),
        )


def test_lagmul():
    x = numpy.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = beignet.orthax.lagval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.orthax.lagval(x, pol2)
            x = beignet.orthax.lagmul(pol1, pol2)
            pol3 = beignet.orthax.lagtrim(x, tol=1e-6)
            val3 = beignet.orthax.lagval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.orthax.lagadd(ci, cj)
            quo, rem = beignet.orthax.lagdiv(tgt, ci)
            res = beignet.orthax.lagadd(beignet.orthax.lagmul(quo, ci), rem)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.orthax.lagmul, [c] * j, numpy.array([1]))
            res = beignet.orthax.lagpow(c, j)
            numpy.testing.assert_array_equal(
                beignet.orthax.lagtrim(res, tol=1e-6),
                beignet.orthax.lagtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_legint():  # noqa:C901
    numpy.testing.assert_raises(TypeError, beignet.orthax.legint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.orthax.legint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.orthax.legint([0], m=i, k=k)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legtrim(res, tol=1e-6), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        legpol = beignet.orthax.poly2leg(pol)
        legint = beignet.orthax.legint(legpol, m=1, k=[i])
        res = beignet.orthax.leg2poly(legint)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legtrim(res, tol=1e-6),
            beignet.orthax.legtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        legpol = beignet.orthax.poly2leg(pol)
        legint = beignet.orthax.legint(legpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_array_almost_equal(beignet.orthax.legval(-1, legint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        legpol = beignet.orthax.poly2leg(pol)
        legint = beignet.orthax.legint(legpol, m=1, k=[i], scl=2)
        res = beignet.orthax.leg2poly(legint)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.legtrim(res, tol=1e-6),
            beignet.orthax.legtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.orthax.legint(tgt, m=1)
            res = beignet.orthax.legint(pol, m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.legint(tgt, m=1, k=[k])
            res = beignet.orthax.legint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.legint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.orthax.legint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.orthax.legint(tgt, m=1, k=[k], scl=2)
            res = beignet.orthax.legint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

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

    numpy.testing.assert_array_equal(beignet.orthax.legint((1, 2, 3), 0), (1, 2, 3))


def test_legder():
    numpy.testing.assert_raises(TypeError, beignet.orthax.legder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.legder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.orthax.legder(tgt, m=0)
        numpy.testing.assert_array_equal(
            beignet.orthax.legtrim(res, tol=1e-6),
            beignet.orthax.legtrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.legder(beignet.orthax.legint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.legder(
                beignet.orthax.legint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.legder(c) for c in c2d.T]).T
    res = beignet.orthax.legder(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.legder(c) for c in c2d])
    res = beignet.orthax.legder(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)

    c = (1, 2, 3, 4)
    numpy.testing.assert_array_equal(beignet.orthax.legder(c, 4), [0])


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
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermefit, [1], [1], (-1,))
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
    numpy.testing.assert_array_almost_equal(beignet.orthax.hermefit(x, x, 1), [0, 1])
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


def test_hermecompanion():
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermecompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermecompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.orthax.hermecompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.orthax.hermecompanion([1, 2])[0, 0] == -0.5)


def test_hermegauss(self):
    x, w = beignet.orthax.hermegauss(100)

    v = beignet.orthax.hermevander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_array_almost_equal(vv, numpy.eye(100))

    tgt = numpy.sqrt(2 * numpy.pi)
    numpy.testing.assert_array_almost_equal(w.sum(), tgt)

    def test_hermeint():  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermeint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], -1)
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], 1, [0, 0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], lbnd=[0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.hermeint, [0], scl=[0])
        numpy.testing.assert_raises(TypeError, beignet.orthax.hermeint, [0], axis=0.5)

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.hermeint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=1e-6), [0, 1]
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermepol = beignet.orthax.poly2herme(pol)
            hermeint = beignet.orthax.hermeint(hermepol, m=1, k=[i])
            res = beignet.orthax.herme2poly(hermeint)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
            )

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
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
            )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1)
                res = beignet.orthax.hermeint(pol, m=j)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermetrim(res, tol=1e-6),
                    beignet.orthax.hermetrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1, k=[k])
                res = beignet.orthax.hermeint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermetrim(res, tol=1e-6),
                    beignet.orthax.hermetrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermetrim(res, tol=1e-6),
                    beignet.orthax.hermetrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.hermeint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.hermeint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.hermetrim(res, tol=1e-6),
                    beignet.orthax.hermetrim(tgt, tol=1e-6),
                )

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


def test_hermeder():
    numpy.testing.assert_raises(TypeError, beignet.orthax.hermeder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.orthax.hermeder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.orthax.hermeder(tgt, m=0)
        numpy.testing.assert_array_equal(
            beignet.orthax.hermetrim(res, tol=1e-6),
            beignet.orthax.hermetrim(tgt, tol=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.hermeder(beignet.orthax.hermeint(tgt, m=j), m=j)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.orthax.hermeder(
                beignet.orthax.hermeint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=1e-6),
                beignet.orthax.hermetrim(tgt, tol=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.orthax.hermeder(c) for c in c2d.T]).T
    res = beignet.orthax.hermeder(c2d, axis=0)
    numpy.testing.assert_array_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.orthax.hermeder(c) for c in c2d])
    res = beignet.orthax.hermeder(c2d, axis=1)
    numpy.testing.assert_array_almost_equal(res, tgt)


def test_hermevander():
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


def test_hermevander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.orthax.hermevander2d(x1, x2, (1, 2))
    tgt = beignet.orthax.hermeval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.hermevander2d([x1], [x2], (1, 2))
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_hermevander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.orthax.hermevander3d(x1, x2, x3, (1, 2, 3))
    tgt = beignet.orthax.hermeval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_array_almost_equal(res, tgt)

    van = beignet.orthax.hermevander3d([x1], [x2], [x3], (1, 2, 3))
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_hermadd():
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


def test_hermsub():
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


def test_hermmulx():
    x = beignet.orthax.hermmulx([0.0])
    numpy.testing.assert_array_equal(beignet.orthax.hermtrim(x, tol=1e-6), [0.0])
    numpy.testing.assert_array_equal(beignet.orthax.hermmulx([1.0]), [0.0, 0.5])
    for i in range(1, 5):
        ser = [0.0] * i + [1.0]
        tgt = [0.0] * (i - 1) + [i, 0.0, 0.5]
        numpy.testing.assert_array_equal(beignet.orthax.hermmulx(ser), tgt)


def test_hermmul():
    x = numpy.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = [0.0] * i + [1.0]
        val1 = beignet.orthax.hermval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0.0] * j + [1.0]
            val2 = beignet.orthax.hermval(x, pol2)
            pol3 = beignet.orthax.hermmul(pol1, pol2)
            val3 = beignet.orthax.hermval(x, pol3)
            numpy.testing.assert_(
                len(beignet.orthax.hermtrim(pol3, tol=1e-6)) == i + j + 1, msg
            )
            numpy.testing.assert_array_almost_equal(val3, val1 * val2, err_msg=msg)


def test_hermdiv():
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


def test_hermpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1).astype(float)
            tgt = functools.reduce(beignet.orthax.hermmul, [c] * j, numpy.array([1]))
            res = beignet.orthax.hermpow(c, j)
            numpy.testing.assert_array_equal(
                beignet.orthax.hermtrim(res, tol=1e-6),
                beignet.orthax.hermtrim(tgt, tol=1e-6),
                err_msg=msg,
            )


def test_hermint():
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
        numpy.testing.assert_array_almost_equal(beignet.orthax.hermval(-1, hermint), i)

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


def test_polyval():
    # c1d = numpy.array([1.0, 2.0, 3.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    # y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

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


def test_polyvalfromroots():
    # c1d = numpy.array([1.0, 2.0, 3.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    # y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_raises(
        ValueError,
        beignet.orthax.polyvalfromroots,
        [1],
        [1],
        tensor=False,
    )

    numpy.testing.assert_equal(beignet.orthax.polyvalfromroots([], [1]).size, 0)
    numpy.testing.assert_(beignet.orthax.polyvalfromroots([], [1]).shape == (0,))

    numpy.testing.assert_equal(beignet.orthax.polyvalfromroots([], [[1] * 5]).size, 0)
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
        numpy.testing.assert_equal(beignet.orthax.polyvalfromroots(x, [1]).shape, dims)
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


def test_polyval2d():
    c1d = numpy.array([1.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises_regex(
        ValueError,
        "incompatible",
        beignet.orthax.polyval2d,
        x1,
        x2[:2],
        c2d,
    )

    tgt = y1 * y2
    res = beignet.orthax.polyval2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.polyval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_polyval3d():
    c1d = numpy.array([1.0, 2.0, 3.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises_regex(
        ValueError,
        "incompatible",
        beignet.orthax.polyval3d,
        x1,
        x2,
        x3[:2],
        c3d,
    )

    tgt = y1 * y2 * y3
    res = beignet.orthax.polyval3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.polyval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_polygrid2d():
    c1d = numpy.array([1.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.orthax.polygrid2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.polygrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_polygrid3d():
    c1d = numpy.array([1.0, 2.0, 3.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.orthax.polygrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.polygrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_legval():
    # c1d = numpy.array([2.0, 2.0, 2.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

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
        numpy.testing.assert_array_equal(beignet.orthax.legval(x, [1, 0]).shape, dims)
        numpy.testing.assert_array_equal(
            beignet.orthax.legval(x, [1, 0, 0]).shape, dims
        )


def test_legval2d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(ValueError, beignet.orthax.legval2d, x1, x2[:2], c2d)

    tgt = y1 * y2
    res = beignet.orthax.legval2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.legval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_legval3d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.legval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.orthax.legval3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.legval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_leggrid2d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.orthax.leggrid2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.leggrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_leggrid3d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.orthax.leggrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.leggrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_chebval():
    numpy.testing.assert_array_equal(beignet.orthax.chebval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [numpy.polynomial.polynomial.polyval(x, c) for c in chebcoefficients]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.orthax.chebval(x, [0] * i + [1])
        numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        numpy.testing.assert_array_equal(beignet.orthax.chebval(x, [1]).shape, dims)
        numpy.testing.assert_array_equal(beignet.orthax.chebval(x, [1, 0]).shape, dims)
        numpy.testing.assert_array_equal(
            beignet.orthax.chebval(x, [1, 0, 0]).shape, dims
        )


def test_chebval2d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(ValueError, beignet.orthax.chebval2d, x1, x2[:2], c2d)

    tgt = y1 * y2
    res = beignet.orthax.chebval2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.chebval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_chebval3d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.chebval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.orthax.chebval3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.chebval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_chebgrid2d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.orthax.chebgrid2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.chebgrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_chebgrid3d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.orthax.chebgrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.chebgrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_hermval():
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


def test_hermval2d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(ValueError, beignet.orthax.hermval2d, x1, x2[:2], c2d)

    tgt = y1 * y2
    res = beignet.orthax.hermval2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermval3d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.hermval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.orthax.hermval3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermgrid2d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.orthax.hermgrid2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermgrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_hermgrid3d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.orthax.hermgrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermgrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_hermeval():
    numpy.testing.assert_array_equal(beignet.orthax.hermeval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [numpy.polynomial.polynomial.polyval(x, c) for c in hermecoefficients]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.orthax.hermeval(x, [0] * i + [1])
        numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        numpy.testing.assert_array_equal(beignet.orthax.hermeval(x, [1]).shape, dims)
        numpy.testing.assert_array_equal(beignet.orthax.hermeval(x, [1, 0]).shape, dims)
        numpy.testing.assert_array_equal(
            beignet.orthax.hermeval(x, [1, 0, 0]).shape, dims
        )


def test_hermeval2d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(ValueError, beignet.orthax.hermeval2d, x1, x2[:2], c2d)

    tgt = y1 * y2
    res = beignet.orthax.hermeval2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermeval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermeval3d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.hermeval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.orthax.hermeval3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermeval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermegrid2d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.orthax.hermegrid2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermegrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_hermegrid3d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.orthax.hermegrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.hermegrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_lagval():
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
        numpy.testing.assert_array_equal(beignet.orthax.lagval(x, [1, 0]).shape, dims)
        numpy.testing.assert_array_equal(
            beignet.orthax.lagval(x, [1, 0, 0]).shape, dims
        )


def test_lagval2d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(ValueError, beignet.orthax.lagval2d, x1, x2[:2], c2d)

    tgt = y1 * y2
    res = beignet.orthax.lagval2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.lagval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_lagval3d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.lagval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.orthax.lagval3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.lagval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_laggrid2d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    # c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.orthax.laggrid2d(x1, x2, c2d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.laggrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_laggrid3d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    # c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.orthax.laggrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_array_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.orthax.laggrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_chebinterpolate():
    def f(self, x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(ValueError, beignet.orthax.chebinterpolate, f, -1)

    for deg in range(1, 5):
        numpy.testing.assert_(
            beignet.orthax.chebinterpolate(f, deg).shape == (deg + 1,)
        )

    def powx(x, p):
        return x**p

    x = numpy.linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = beignet.orthax.chebinterpolate(powx, deg, (p,))
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.chebval(x, c), powx(x, p), decimal=12
            )
