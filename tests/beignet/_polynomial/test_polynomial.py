import functools

import beignet.polynomial
import numpy
import numpy.testing
import torch

chebyshev_polynomial_T0 = [1]
chebyshev_polynomial_T1 = [0, 1]
chebyshev_polynomial_T2 = [-1, 0, 2]
chebyshev_polynomial_T3 = [0, -3, 0, 4]
chebyshev_polynomial_T4 = [1, 0, -8, 0, 8]
chebyshev_polynomial_T5 = [0, 5, 0, -20, 0, 16]
chebyshev_polynomial_T6 = [-1, 0, 18, 0, -48, 0, 32]
chebyshev_polynomial_T7 = [0, -7, 0, 56, 0, -112, 0, 64]
chebyshev_polynomial_T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
chebyshev_polynomial_T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

chebyshev_polynomial_Tlist = [
    chebyshev_polynomial_T0,
    chebyshev_polynomial_T1,
    chebyshev_polynomial_T2,
    chebyshev_polynomial_T3,
    chebyshev_polynomial_T4,
    chebyshev_polynomial_T5,
    chebyshev_polynomial_T6,
    chebyshev_polynomial_T7,
    chebyshev_polynomial_T8,
    chebyshev_polynomial_T9,
]

hermite_polynomial_H0 = numpy.array([1])
hermite_polynomial_H1 = numpy.array([0, 2])
hermite_polynomial_H2 = numpy.array([-2, 0, 4])
hermite_polynomial_H3 = numpy.array([0, -12, 0, 8])
hermite_polynomial_H4 = numpy.array([12, 0, -48, 0, 16])
hermite_polynomial_H5 = numpy.array([0, 120, 0, -160, 0, 32])
hermite_polynomial_H6 = numpy.array([-120, 0, 720, 0, -480, 0, 64])
hermite_polynomial_H7 = numpy.array([0, -1680, 0, 3360, 0, -1344, 0, 128])
hermite_polynomial_H8 = numpy.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
hermite_polynomial_H9 = numpy.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])

hermite_polynomial_Hlist = [
    hermite_polynomial_H0,
    hermite_polynomial_H1,
    hermite_polynomial_H2,
    hermite_polynomial_H3,
    hermite_polynomial_H4,
    hermite_polynomial_H5,
    hermite_polynomial_H6,
    hermite_polynomial_H7,
    hermite_polynomial_H8,
    hermite_polynomial_H9,
]

hermite_e_polynomial_He0 = numpy.array([1])
hermite_e_polynomial_He1 = numpy.array([0, 1])
hermite_e_polynomial_He2 = numpy.array([-1, 0, 1])
hermite_e_polynomial_He3 = numpy.array([0, -3, 0, 1])
hermite_e_polynomial_He4 = numpy.array([3, 0, -6, 0, 1])
hermite_e_polynomial_He5 = numpy.array([0, 15, 0, -10, 0, 1])
hermite_e_polynomial_He6 = numpy.array([-15, 0, 45, 0, -15, 0, 1])
hermite_e_polynomial_He7 = numpy.array([0, -105, 0, 105, 0, -21, 0, 1])
hermite_e_polynomial_He8 = numpy.array([105, 0, -420, 0, 210, 0, -28, 0, 1])
hermite_e_polynomial_He9 = numpy.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])

hermite_e_polynomial_Helist = [
    hermite_e_polynomial_He0,
    hermite_e_polynomial_He1,
    hermite_e_polynomial_He2,
    hermite_e_polynomial_He3,
    hermite_e_polynomial_He4,
    hermite_e_polynomial_He5,
    hermite_e_polynomial_He6,
    hermite_e_polynomial_He7,
    hermite_e_polynomial_He8,
    hermite_e_polynomial_He9,
]

laguerre_polynomial_L0 = numpy.array([1]) / 1
laguerre_polynomial_L1 = numpy.array([1, -1]) / 1
laguerre_polynomial_L2 = numpy.array([2, -4, 1]) / 2
laguerre_polynomial_L3 = numpy.array([6, -18, 9, -1]) / 6
laguerre_polynomial_L4 = numpy.array([24, -96, 72, -16, 1]) / 24
laguerre_polynomial_L5 = numpy.array([120, -600, 600, -200, 25, -1]) / 120
laguerre_polynomial_L6 = numpy.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720

laguerre_polynomial_Llist = [
    laguerre_polynomial_L0,
    laguerre_polynomial_L1,
    laguerre_polynomial_L2,
    laguerre_polynomial_L3,
    laguerre_polynomial_L4,
    laguerre_polynomial_L5,
    laguerre_polynomial_L6,
]

legendre_polynomial_L0 = numpy.array([1])
legendre_polynomial_L1 = numpy.array([0, 1])
legendre_polynomial_L2 = numpy.array([-1, 0, 3]) / 2
legendre_polynomial_L3 = numpy.array([0, -3, 0, 5]) / 2
legendre_polynomial_L4 = numpy.array([3, 0, -30, 0, 35]) / 8
legendre_polynomial_L5 = numpy.array([0, 15, 0, -70, 0, 63]) / 8
legendre_polynomial_L6 = numpy.array([-5, 0, 105, 0, -315, 0, 231]) / 16
legendre_polynomial_L7 = numpy.array([0, -35, 0, 315, 0, -693, 0, 429]) / 16
legendre_polynomial_L8 = numpy.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128
legendre_polynomial_L9 = (
    numpy.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128
)

legendre_polynomial_Llist = [
    legendre_polynomial_L0,
    legendre_polynomial_L1,
    legendre_polynomial_L2,
    legendre_polynomial_L3,
    legendre_polynomial_L4,
    legendre_polynomial_L5,
    legendre_polynomial_L6,
    legendre_polynomial_L7,
    legendre_polynomial_L8,
    legendre_polynomial_L9,
]

polynomial_T0 = [1]
polynomial_T1 = [0, 1]
polynomial_T2 = [-1, 0, 2]
polynomial_T3 = [0, -3, 0, 4]
polynomial_T4 = [1, 0, -8, 0, 8]
polynomial_T5 = [0, 5, 0, -20, 0, 16]
polynomial_T6 = [-1, 0, 18, 0, -48, 0, 32]
polynomial_T7 = [0, -7, 0, 56, 0, -112, 0, 64]
polynomial_T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
polynomial_T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

polynomial_Tlist = [
    polynomial_T0,
    polynomial_T1,
    polynomial_T2,
    polynomial_T3,
    polynomial_T4,
    polynomial_T5,
    polynomial_T6,
    polynomial_T7,
    polynomial_T8,
    polynomial_T9,
]


def test_cheb2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.cheb2poly([0] * i + [1]),
            chebyshev_polynomial_Tlist[i],
        )


def test_chebadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebadd([0] * i + [1], [0] * j + [1]),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_chebcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebcompanion, [1])

    for i in range(1, 5):
        numpy.testing.assert_(
            beignet.polynomial.chebcompanion([0] * i + [1]).shape == (i, i)
        )

    numpy.testing.assert_(beignet.polynomial.chebcompanion([1, 2])[0, 0] == -0.5)


def test_chebder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial.chebtrim(
                beignet.polynomial.chebder(tgt, m=0), tolerance=1e-6
            ),
            beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebder(
                        beignet.polynomial.chebint(tgt, m=j), m=j
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebder(
                        beignet.polynomial.chebint(tgt, m=j, scl=2), m=j, scl=0.5
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebder(c2d, axis=0),
        numpy.vstack([beignet.polynomial.chebder(c) for c in c2d.T]).T,
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebder(c2d, axis=1),
        numpy.vstack([beignet.polynomial.chebder(c) for c in c2d]),
    )


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            tgt = beignet.polynomial.chebadd(ci, [0] * j + [1])
            quo, rem = beignet.polynomial.chebdiv(tgt, ci)
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebadd(
                        beignet.polynomial.chebmul(quo, ci), rem
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_chebdomain():
    torch.testing.assert_close(beignet.polynomial.chebdomain, [-1, 1])


def test_chebfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.chebfit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.chebfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.chebfit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.chebfit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebfit, [1], [1], [])

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial.chebfit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef3), y)
    coef3 = beignet.polynomial.chebfit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef3), y)

    coef4 = beignet.polynomial.chebfit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef4), y)
    coef4 = beignet.polynomial.chebfit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef4), y)

    coef4 = beignet.polynomial.chebfit(x, y, [2, 3, 4, 1, 0])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef4), y)

    coef2d = beignet.polynomial.chebfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.chebfit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial.chebfit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial.chebfit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial.chebfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.chebfit(x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(beignet.polynomial.chebfit(x, x, 1), [0, 1])
    numpy.testing.assert_almost_equal(beignet.polynomial.chebfit(x, x, [0, 1]), [0, 1])

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.polynomial.chebfit(x, y, 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef1), y)
    coef2 = beignet.polynomial.chebfit(x, y, [0, 2, 4])
    numpy.testing.assert_almost_equal(beignet.polynomial.chebval(x, coef2), y)
    numpy.testing.assert_almost_equal(coef1, coef2)


def test_chebfromroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebtrim(
            beignet.polynomial.chebfromroots([]), tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        tgt = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            beignet.polynomial.chebtrim(
                beignet.polynomial.chebfromroots(roots) * 2 ** (i - 1),
                tolerance=1e-6,
            ),
            beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
        )


def test_chebgauss():
    x, w = beignet.polynomial.chebgauss(100)

    v = beignet.polynomial.chebvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    numpy.testing.assert_almost_equal(vd[:, None] * vv * vd, numpy.eye(100))

    numpy.testing.assert_almost_equal(numpy.sum(w), numpy.pi)


def test_chebgrid2d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebgrid2d(x1, x2, c2d), numpy.einsum("i,j->ij", y1, y2)
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(beignet.polynomial.chebgrid2d(z, z, c2d).shape == (2, 3) * 2)


def test_chebgrid3d():
    c1d = numpy.array([2.5, 2.0, 1.5])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebgrid3d(x1, x2, x3, c3d), tgt
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial.chebgrid3d(z, z, z, c3d).shape == (2, 3) * 3
    )


def test_chebint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.chebint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.chebint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        chebpol = beignet.polynomial.poly2cheb(pol)
        chebint = beignet.polynomial.chebint(chebpol, m=1, k=[i])
        res = beignet.polynomial.cheb2poly(chebint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.chebtrim(res, tolerance=1e-6),
            beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        chebpol = beignet.polynomial.poly2cheb(pol)
        chebint = beignet.polynomial.chebint(chebpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.chebval(-1, chebint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        chebpol = beignet.polynomial.poly2cheb(pol)
        chebint = beignet.polynomial.chebint(chebpol, m=1, k=[i], scl=2)
        res = beignet.polynomial.cheb2poly(chebint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.chebtrim(res, tolerance=1e-6),
            beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1)
            res = beignet.polynomial.chebint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1, k=[k])
            res = beignet.polynomial.chebint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.chebint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.chebint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.chebint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.chebint(c) for c in c2d.T]).T
    res = beignet.polynomial.chebint(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.chebint(c) for c in c2d])
    res = beignet.polynomial.chebint(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.chebint(c, k=3) for c in c2d])
    res = beignet.polynomial.chebint(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)


def test_chebinterpolate():
    def func(x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.chebinterpolate, func, -1
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.chebinterpolate, func, 10.0
    )

    for deg in range(1, 5):
        numpy.testing.assert_(
            beignet.polynomial.chebinterpolate(func, deg).shape == (deg + 1,)
        )

    def powx(x, p):
        return x**p

    x = numpy.linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = beignet.polynomial.chebinterpolate(powx, deg, (p,))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.chebval(x, c), powx(x, p), decimal=12
            )


def test_chebline():
    torch.testing.assert_close(beignet.polynomial.chebline(3, 4), [3, 4])


def test_chebmul():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(i + j + 1)
            tgt[i + j] += 0.5
            tgt[abs(i - j)] += 0.5
            res = beignet.polynomial.chebmul([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_chebmulx():
    torch.testing.assert_close(beignet.polynomial.chebmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.chebmulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [0.5, 0, 0.5]
        torch.testing.assert_close(beignet.polynomial.chebmulx(ser), tgt)


def test_chebone():
    torch.testing.assert_close(beignet.polynomial.chebone, [1])


def test_chebpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.chebmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial.chebpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_chebpts1():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebpts1, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebpts1, 0)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebpts1(1), [0])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts1(2), [-0.70710678118654746, 0.70710678118654746]
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts1(3), [-0.86602540378443871, 0, 0.86602540378443871]
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts1(4),
        [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325],
    )


def test_chebpts2():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebpts2, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebpts2, 1)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebpts2(2), [-1, 1])
    numpy.testing.assert_almost_equal(beignet.polynomial.chebpts2(3), [-1, 0, 1])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts2(4), [-1, -0.5, 0.5, 1]
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts2(5), [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
    )


def test_chebroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.chebroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.chebroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.chebtrim(
                beignet.polynomial.chebroots(beignet.polynomial.chebfromroots(tgt)),
                tolerance=1e-6,
            ),
            beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
        )


def test_chebsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.polynomial.chebsub([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(res, tolerance=1e-6),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_chebtrim():
    coef = [2, -1, 1, 0]
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebtrim, coef, -1)
    torch.testing.assert_close(beignet.polynomial.chebtrim(coef), coef[:-1])
    torch.testing.assert_close(beignet.polynomial.chebtrim(coef, 1), coef[:-3])
    torch.testing.assert_close(beignet.polynomial.chebtrim(coef, 2), [0])


def test_chebval():
    torch.testing.assert_close(beignet.polynomial.chebval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial.polyval(x, c) for c in chebyshev_polynomial_Tlist]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.chebval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(beignet.polynomial.chebval(x, [1]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.chebval(x, [1, 0]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.chebval(x, [1, 0, 0]).shape, dims)


def test_chebval2d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.chebval2d, x1, x2[:2], c2d
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebval2d(x1, x2, c2d), y1 * y2
    )
    z = numpy.ones((2, 3))
    numpy.testing.assert_(beignet.polynomial.chebval2d(z, z, c2d).shape == (2, 3))


def test_chebval3d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)
    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.chebval3d, x1, x2, x3[:2], c3d
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebval3d(x1, x2, x3, c3d), y1 * y2 * y3
    )
    z = numpy.ones((2, 3))
    numpy.testing.assert_(beignet.polynomial.chebval3d(z, z, z, c3d).shape == (2, 3))


def test_chebvander():
    x = numpy.arange(3)
    v = beignet.polynomial.chebvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.chebval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial.chebvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.chebval(x, coef)
        )


def test_chebvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial.chebvander2d(x1, x2, [1, 2])
    tgt = beignet.polynomial.chebval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.chebvander2d([x1], [x2], [1, 2])
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_chebvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    numpy.testing.assert_almost_equal(
        numpy.dot(beignet.polynomial.chebvander3d(x1, x2, x3, [1, 2, 3]), c.flat),
        beignet.polynomial.chebval3d(x1, x2, x3, c),
    )
    numpy.testing.assert_(
        beignet.polynomial.chebvander3d([x1], [x2], [x3], [1, 2, 3]).shape == (1, 5, 24)
    )


def test_chebweight():
    x = numpy.linspace(-1, 1, 11)[1:-1]
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebweight(x), 1.0 / (numpy.sqrt(1 + x) * numpy.sqrt(1 - x))
    )


def test_chebx():
    torch.testing.assert_close(beignet.polynomial.chebx, [0, 1])


def test_chebzero():
    torch.testing.assert_close(beignet.polynomial.chebzero, [0])


def test_herm2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.herm2poly([0] * i + [1]),
            hermite_polynomial_Hlist[i],
        )


def test_hermadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.polynomial.hermadd([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.hermcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.hermcompanion([1, 2])[0, 0] == -0.25)


def test_hermder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial.hermtrim(
                beignet.polynomial.hermder(tgt, m=0), tolerance=1e-6
            ),
            beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermder(
                        beignet.polynomial.hermint(tgt, m=j), m=j
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermder(
                        beignet.polynomial.hermint(tgt, m=j, scl=2), m=j, scl=0.5
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.hermder(c) for c in c2d.T]).T
    numpy.testing.assert_almost_equal(beignet.polynomial.hermder(c2d, axis=0), tgt)

    tgt = numpy.vstack([beignet.polynomial.hermder(c) for c in c2d])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermder(c2d, axis=1), tgt)


def test_hermdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.hermadd(ci, cj)
            quo, rem = beignet.polynomial.hermdiv(tgt, ci)
            res = beignet.polynomial.hermadd(beignet.polynomial.hermmul(quo, ci), rem)
            torch.testing.assert_close(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermdomain():
    torch.testing.assert_close(beignet.polynomial.hermdomain, [-1, 1])


def test_herme2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.herme2poly([0] * i + [1]),
            hermite_e_polynomial_Helist[i],
        )


def test_hermeadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.polynomial.hermeadd([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermecompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermecompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermecompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.hermecompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.hermecompanion([1, 2])[0, 0] == -0.5)


def test_hermeder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermeder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial.hermeder(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial.hermetrim(res, tolerance=1e-6),
            beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.hermeder(
                beignet.polynomial.hermeint(tgt, m=j), m=j
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.hermeder(
                beignet.polynomial.hermeint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.hermeder(c) for c in c2d.T]).T
    res = beignet.polynomial.hermeder(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermeder(c) for c in c2d])
    res = beignet.polynomial.hermeder(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)


def test_hermediv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.hermeadd(ci, cj)
            quo, rem = beignet.polynomial.hermediv(tgt, ci)
            res = beignet.polynomial.hermeadd(beignet.polynomial.hermemul(quo, ci), rem)
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermedomain():
    torch.testing.assert_close(beignet.polynomial.hermedomain, [-1, 1])


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermefit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1], [1, 2], 0)
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.hermefit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.hermefit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.hermefit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.hermefit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1], [1], [])

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial.hermefit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef3), y)
    coef3 = beignet.polynomial.hermefit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef3), y)

    coef4 = beignet.polynomial.hermefit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef4), y)
    coef4 = beignet.polynomial.hermefit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef4), y)

    coef4 = beignet.polynomial.hermefit(x, y, [2, 3, 4, 1, 0])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef4), y)

    coef2d = beignet.polynomial.hermefit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.hermefit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial.hermefit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial.hermefit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial.hermefit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.hermefit(x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(beignet.polynomial.hermefit(x, x, 1), [0, 1])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermefit(x, x, [0, 1]), [0, 1])

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.polynomial.hermefit(x, y, 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef1), y)
    coef2 = beignet.polynomial.hermefit(x, y, [0, 2, 4])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef2), y)
    numpy.testing.assert_almost_equal(coef1, coef2)


def test_hermefromroots():
    res = beignet.polynomial.hermefromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.hermetrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.hermefromroots(roots)
        res = beignet.polynomial.hermeval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.herme2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(res, tgt)


def test_hermegauss():
    x, w = beignet.polynomial.hermegauss(100)

    v = beignet.polynomial.hermevander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))

    tgt = numpy.sqrt(2 * numpy.pi)
    numpy.testing.assert_almost_equal(w.sum(), tgt)


def test_hermegrid2d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.polynomial.hermegrid2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermegrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_hermegrid3d():
    c1d = numpy.array([4.0, 2.0, 3.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.polynomial.hermegrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermegrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_hermeint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermeint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermeint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermeint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.hermeint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 1])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermepol = beignet.polynomial.poly2herme(pol)
        hermeint = beignet.polynomial.hermeint(hermepol, m=1, k=[i])
        res = beignet.polynomial.herme2poly(hermeint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermetrim(res, tolerance=1e-6),
            beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermepol = beignet.polynomial.poly2herme(pol)
        hermeint = beignet.polynomial.hermeint(hermepol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(-1, hermeint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermepol = beignet.polynomial.poly2herme(pol)
        hermeint = beignet.polynomial.hermeint(hermepol, m=1, k=[i], scl=2)
        res = beignet.polynomial.herme2poly(hermeint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermetrim(res, tolerance=1e-6),
            beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1)
            res = beignet.polynomial.hermeint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1, k=[k])
            res = beignet.polynomial.hermeint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermeint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.hermeint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.hermeint(c) for c in c2d.T]).T
    res = beignet.polynomial.hermeint(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermeint(c) for c in c2d])
    res = beignet.polynomial.hermeint(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermeint(c, k=3) for c in c2d])
    res = beignet.polynomial.hermeint(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)


def test_hermeline():
    torch.testing.assert_close(beignet.polynomial.hermeline(3, 4), [3, 4])


def test_hermemul():
    x = numpy.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = beignet.polynomial.hermeval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.polynomial.hermeval(x, pol2)
            pol3 = beignet.polynomial.hermemul(pol1, pol2)
            val3 = beignet.polynomial.hermeval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)


def test_hermemulx():
    torch.testing.assert_close(beignet.polynomial.hermemulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.hermemulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i, 0, 1]
        torch.testing.assert_close(beignet.polynomial.hermemulx(ser), tgt)


def test_hermeone():
    torch.testing.assert_close(beignet.polynomial.hermeone, [1])


def test_hermepow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.hermemul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial.hermepow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermeroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeroots([1, 1]), [-1])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial.hermeroots(beignet.polynomial.hermefromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermetrim(res, tolerance=1e-6),
            beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
        )


def test_hermesub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.polynomial.hermesub([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermetrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermetrim, coef, -1)

    torch.testing.assert_close(beignet.polynomial.hermetrim(coef), coef[:-1])
    torch.testing.assert_close(beignet.polynomial.hermetrim(coef, 1), coef[:-3])
    torch.testing.assert_close(beignet.polynomial.hermetrim(coef, 2), [0])


def test_hermeval():
    x = numpy.random.random((3, 5)) * 2 - 1

    torch.testing.assert_close(beignet.polynomial.hermeval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial.polyval(x, c) for c in hermite_e_polynomial_Helist]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.hermeval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(beignet.polynomial.hermeval(x, [1]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.hermeval(x, [1, 0]).shape, dims)
        torch.testing.assert_close(
            beignet.polynomial.hermeval(x, [1, 0, 0]).shape, dims
        )


def test_hermeval2d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.hermeval2d, x1, x2[:2], c2d
    )

    tgt = y1 * y2
    res = beignet.polynomial.hermeval2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermeval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermeval3d():
    c1d = numpy.array([4.0, 2.0, 3.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.hermeval3d,
        x1,
        x2,
        x3[:2],
        c3d,
    )

    tgt = y1 * y2 * y3
    res = beignet.polynomial.hermeval3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermeval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermevander():
    x = numpy.arange(3)
    v = beignet.polynomial.hermevander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.hermeval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial.hermevander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.hermeval(x, coef)
        )


def test_hermevander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial.hermevander2d(x1, x2, [1, 2])
    tgt = beignet.polynomial.hermeval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.hermevander2d([x1], [x2], [1, 2])
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_hermevander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.polynomial.hermevander3d(x1, x2, x3, [1, 2, 3])
    tgt = beignet.polynomial.hermeval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.hermevander3d([x1], [x2], [x3], [1, 2, 3])
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_hermeweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-0.5 * x**2)
    res = beignet.polynomial.hermeweight(x)
    numpy.testing.assert_almost_equal(res, tgt)


def test_hermex():
    torch.testing.assert_close(beignet.polynomial.hermex, [0, 1])


def test_hermezero():
    torch.testing.assert_close(beignet.polynomial.hermezero, [0])


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.hermfit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.hermfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.hermfit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.hermfit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermfit, [1], [1], [])

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial.hermfit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef3), y)
    coef3 = beignet.polynomial.hermfit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef3), y)

    coef4 = beignet.polynomial.hermfit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef4), y)
    coef4 = beignet.polynomial.hermfit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef4), y)

    coef4 = beignet.polynomial.hermfit(x, y, [2, 3, 4, 1, 0])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef4), y)

    coef2d = beignet.polynomial.hermfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.hermfit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial.hermfit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial.hermfit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial.hermfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.hermfit(x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(beignet.polynomial.hermfit(x, x, 1), [0, 0.5])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.hermfit(x, x, [0, 1]), [0, 0.5]
    )

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.polynomial.hermfit(x, y, 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef1), y)
    coef2 = beignet.polynomial.hermfit(x, y, [0, 2, 4])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermval(x, coef2), y)
    numpy.testing.assert_almost_equal(coef1, coef2)


def test_hermfromroots():
    res = beignet.polynomial.hermfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.hermtrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.hermfromroots(roots)
        res = beignet.polynomial.hermval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.herm2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(res, tgt)


def test_hermgauss():
    x, w = beignet.polynomial.hermgauss(100)

    v = beignet.polynomial.hermvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))

    tgt = numpy.sqrt(numpy.pi)
    numpy.testing.assert_almost_equal(w.sum(), tgt)


def test_hermgrid2d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.polynomial.hermgrid2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermgrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_hermgrid3d():
    c1d = numpy.array([2.5, 1.0, 0.75])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.polynomial.hermgrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermgrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_hermint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.polynomial.hermint([0], m=i, k=k)
        numpy.testing.assert_almost_equal(res, [0, 0.5])

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermpol = beignet.polynomial.poly2herm(pol)
        hermint = beignet.polynomial.hermint(hermpol, m=1, k=[i])
        res = beignet.polynomial.herm2poly(hermint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermtrim(res, tolerance=1e-6),
            beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermpol = beignet.polynomial.poly2herm(pol)
        hermint = beignet.polynomial.hermint(hermpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.hermval(-1, hermint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermpol = beignet.polynomial.poly2herm(pol)
        hermint = beignet.polynomial.hermint(hermpol, m=1, k=[i], scl=2)
        res = beignet.polynomial.herm2poly(hermint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermtrim(res, tolerance=1e-6),
            beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.hermint(tgt, m=1)
            res = beignet.polynomial.hermint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermint(tgt, m=1, k=[k])
            res = beignet.polynomial.hermint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.hermint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.hermint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.hermint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.hermint(c) for c in c2d.T]).T
    res = beignet.polynomial.hermint(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermint(c) for c in c2d])
    res = beignet.polynomial.hermint(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.hermint(c, k=3) for c in c2d])
    res = beignet.polynomial.hermint(c2d, k=3, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)


def test_hermline():
    torch.testing.assert_close(beignet.polynomial.hermline(3, 4), [3, 2])


def test_hermmul():
    x = numpy.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = beignet.polynomial.hermval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.polynomial.hermval(x, pol2)
            pol3 = beignet.polynomial.hermmul(pol1, pol2)
            val3 = beignet.polynomial.hermval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)


def test_hermmulx():
    torch.testing.assert_close(beignet.polynomial.hermmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.hermmulx([1]), [0, 0.5])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i, 0, 0.5]
        torch.testing.assert_close(beignet.polynomial.hermmulx(ser), tgt)


def test_hermone():
    torch.testing.assert_close(beignet.polynomial.hermone, [1])


def test_hermpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.hermmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial.hermpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.hermroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermroots([1, 1]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial.hermroots(beignet.polynomial.hermfromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermtrim(res, tolerance=1e-6),
            beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
        )


def test_hermsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.polynomial.hermsub([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_hermtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermtrim, coef, -1)

    torch.testing.assert_close(beignet.polynomial.hermtrim(coef), coef[:-1])
    torch.testing.assert_close(beignet.polynomial.hermtrim(coef, 1), coef[:-3])
    torch.testing.assert_close(beignet.polynomial.hermtrim(coef, 2), [0])


def test_hermval():
    torch.testing.assert_close(beignet.polynomial.hermval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial.polyval(x, c) for c in hermite_polynomial_Hlist]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.hermval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(beignet.polynomial.hermval(x, [1]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.hermval(x, [1, 0]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.hermval(x, [1, 0, 0]).shape, dims)


def test_hermval2d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.hermval2d, x1, x2[:2], c2d
    )

    tgt = y1 * y2
    res = beignet.polynomial.hermval2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermval3d():
    c1d = numpy.array([2.5, 1.0, 0.75])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.hermval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.polynomial.hermval3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.hermval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_hermvander():
    x = numpy.arange(3)
    v = beignet.polynomial.hermvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.hermval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial.hermvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.hermval(x, coef)
        )


def test_hermvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial.hermvander2d(x1, x2, [1, 2])
    tgt = beignet.polynomial.hermval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.hermvander2d([x1], [x2], [1, 2])
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_hermvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.polynomial.hermvander3d(x1, x2, x3, [1, 2, 3])
    tgt = beignet.polynomial.hermval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.hermvander3d([x1], [x2], [x3], [1, 2, 3])
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_hermweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-(x**2))
    res = beignet.polynomial.hermweight(x)
    numpy.testing.assert_almost_equal(res, tgt)


def test_hermx():
    torch.testing.assert_close(beignet.polynomial.hermx, [0, 0.5])


def test_hermzero():
    torch.testing.assert_close(beignet.polynomial.hermzero, [0])


def test_lag2poly():
    for i in range(7):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lag2poly([0] * i + [1]), laguerre_polynomial_Llist[i]
        )


def test_lagadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.polynomial.lagadd([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_lagcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.lagcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.lagcompanion([1, 2])[0, 0] == 1.5)


def test_lagder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial.lagder(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial.lagtrim(res, tolerance=1e-6),
            beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.lagder(beignet.polynomial.lagint(tgt, m=j), m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.lagder(
                beignet.polynomial.lagint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.lagder(c) for c in c2d.T]).T
    res = beignet.polynomial.lagder(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.lagder(c) for c in c2d])
    res = beignet.polynomial.lagder(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.lagadd(ci, cj)
            quo, rem = beignet.polynomial.lagdiv(tgt, ci)
            res = beignet.polynomial.lagadd(beignet.polynomial.lagmul(quo, ci), rem)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_lagdomain():
    torch.testing.assert_close(beignet.polynomial.lagdomain, [0, 1])


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.lagfit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.lagfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.lagfit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.lagfit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagfit, [1], [1], [])

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial.lagfit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.lagval(x, coef3), y)
    coef3 = beignet.polynomial.lagfit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.lagval(x, coef3), y)

    coef4 = beignet.polynomial.lagfit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.lagval(x, coef4), y)
    coef4 = beignet.polynomial.lagfit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.lagval(x, coef4), y)

    coef2d = beignet.polynomial.lagfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.lagfit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial.lagfit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial.lagfit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial.lagfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.lagfit(x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(beignet.polynomial.lagfit(x, x, 1), [1, -1])
    numpy.testing.assert_almost_equal(beignet.polynomial.lagfit(x, x, [0, 1]), [1, -1])


def test_lagfromroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagtrim(beignet.polynomial.lagfromroots([]), tolerance=1e-6),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.lagfromroots(roots)
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.lag2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.lagval(roots, pol), 0)


def test_laggauss():
    x, w = beignet.polynomial.laggauss(100)

    v = beignet.polynomial.lagvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))
    numpy.testing.assert_almost_equal(w.sum(), 1.0)


def test_laggrid2d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_almost_equal(
        beignet.polynomial.laggrid2d(x1, x2, c2d), numpy.einsum("i,j->ij", y1, y2)
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(beignet.polynomial.laggrid2d(z, z, c2d).shape == (2, 3) * 2)


def test_laggrid3d():
    c1d = numpy.array([9.0, -14.0, 6.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_almost_equal(
        beignet.polynomial.laggrid3d(x1, x2, x3, c3d),
        numpy.einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial.laggrid3d(z, z, z, c3d).shape == (2, 3) * 3
    )


def test_lagint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.lagint, [0], axis=0.5)

    for i in range(2, 5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagint([0], m=i, k=([0] * (i - 2) + [1])), [1, -1]
        )

    for i in range(5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagtrim(
                beignet.polynomial.lag2poly(
                    beignet.polynomial.lagint(
                        beignet.polynomial.poly2lag([0] * i + [1]), m=1, k=[i]
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.lagtrim([i] + [0] * i + [1 / (i + 1)], tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagval(
                -1,
                beignet.polynomial.lagint(
                    beignet.polynomial.poly2lag([0] * i + [1]), m=1, k=[i], lbnd=-1
                ),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagtrim(
                beignet.polynomial.lag2poly(
                    beignet.polynomial.lagint(
                        beignet.polynomial.poly2lag([0] * i + [1]), m=1, k=[i], scl=2
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.lagtrim([i] + [0] * i + [2 / scl], tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1)
            res = beignet.polynomial.lagint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1, k=[k])
            res = beignet.polynomial.lagint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.lagint(tgt, m=1, k=[k], scl=2)
            res = beignet.polynomial.lagint(pol, m=j, k=list(range(j)), scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagint(c2d, axis=0),
        numpy.vstack([beignet.polynomial.lagint(c) for c in c2d.T]).T,
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagint(c2d, axis=1),
        numpy.vstack([beignet.polynomial.lagint(c) for c in c2d]),
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagint(c2d, k=3, axis=1),
        numpy.vstack([beignet.polynomial.lagint(c, k=3) for c in c2d]),
    )


def test_lagline():
    torch.testing.assert_close(beignet.polynomial.lagline(3, 4), [7, -4])


def test_lagmul():
    x = numpy.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = beignet.polynomial.lagval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.polynomial.lagval(x, pol2)
            pol3 = beignet.polynomial.lagmul(pol1, pol2)
            val3 = beignet.polynomial.lagval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)


def test_lagmulx():
    torch.testing.assert_close(beignet.polynomial.lagmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.lagmulx([1]), [1, -1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
        numpy.testing.assert_almost_equal(beignet.polynomial.lagmulx(ser), tgt)


def test_lagone():
    torch.testing.assert_close(beignet.polynomial.lagone, [1])


def test_lagpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.polynomial.lagmul, [c] * j, numpy.array([1]))
            res = beignet.polynomial.lagpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_lagroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.lagroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.lagroots([0, 1]), [1])
    for i in range(2, 5):
        tgt = numpy.linspace(0, 3, i)
        res = beignet.polynomial.lagroots(beignet.polynomial.lagfromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial.lagtrim(res, tolerance=1e-6),
            beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
        )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.polynomial.lagsub([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_lagtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagtrim, coef, -1)

    torch.testing.assert_close(beignet.polynomial.lagtrim(coef), coef[:-1])
    torch.testing.assert_close(beignet.polynomial.lagtrim(coef, 1), coef[:-3])
    torch.testing.assert_close(beignet.polynomial.lagtrim(coef, 2), [0])


def test_lagval():
    x = numpy.random.random((3, 5)) * 2 - 1

    torch.testing.assert_close(beignet.polynomial.lagval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial.polyval(x, c) for c in laguerre_polynomial_Llist]
    for i in range(7):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.lagval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(beignet.polynomial.lagval(x, [1]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.lagval(x, [1, 0]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.lagval(x, [1, 0, 0]).shape, dims)


def test_lagval2d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.lagval2d, x1, x2[:2], c2d
    )

    tgt = y1 * y2
    res = beignet.polynomial.lagval2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.lagval2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_lagval3d():
    c1d = numpy.array([9.0, -14.0, 6.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.lagval3d, x1, x2, x3[:2], c3d
    )

    tgt = y1 * y2 * y3
    res = beignet.polynomial.lagval3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.lagval3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))


def test_lagvander():
    x = numpy.arange(3)
    v = beignet.polynomial.lagvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(v[..., i], beignet.polynomial.lagval(x, coef))

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial.lagvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(v[..., i], beignet.polynomial.lagval(x, coef))


def test_lagvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial.lagvander2d(x1, x2, [1, 2])
    tgt = beignet.polynomial.lagval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.lagvander2d([x1], [x2], [1, 2])
    numpy.testing.assert_(van.shape == (1, 5, 6))


def test_lagvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.polynomial.lagvander3d(x1, x2, x3, [1, 2, 3])
    tgt = beignet.polynomial.lagval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial.lagvander3d([x1], [x2], [x3], [1, 2, 3])
    numpy.testing.assert_(van.shape == (1, 5, 24))


def test_lagweight():
    x = numpy.linspace(0, 10, 11)
    tgt = numpy.exp(-x)
    res = beignet.polynomial.lagweight(x)
    numpy.testing.assert_almost_equal(res, tgt)


def test_lagx():
    torch.testing.assert_close(beignet.polynomial.lagx, [1, -1])


def test_lagzero():
    torch.testing.assert_close(beignet.polynomial.lagzero, [0])


def test_leg2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.leg2poly([0] * i + [1]), legendre_polynomial_Llist[i]
        )


def test_legadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.polynomial.legadd([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_legcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.legcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.legcompanion([1, 2])[0, 0] == -0.5)


def test_legder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial.legder(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial.legtrim(res, tolerance=1e-6),
            beignet.polynomial.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.legder(beignet.polynomial.legint(tgt, m=j), m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.legder(
                beignet.polynomial.legint(tgt, m=j, scl=2), m=j, scl=0.5
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.legder(c) for c in c2d.T]).T
    res = beignet.polynomial.legder(c2d, axis=0)
    numpy.testing.assert_almost_equal(res, tgt)

    tgt = numpy.vstack([beignet.polynomial.legder(c) for c in c2d])
    res = beignet.polynomial.legder(c2d, axis=1)
    numpy.testing.assert_almost_equal(res, tgt)

    c = (1, 2, 3, 4)
    torch.testing.assert_close(beignet.polynomial.legder(c, 4), [0])


def test_legdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.legadd(ci, cj)
            quo, rem = beignet.polynomial.legdiv(tgt, ci)
            res = beignet.polynomial.legadd(beignet.polynomial.legmul(quo, ci), rem)
            torch.testing.assert_close(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )


def test_legdomain():
    torch.testing.assert_close(beignet.polynomial.legdomain, [-1, 1])


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.polynomial.legfit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legfit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legfit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legfit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legfit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legfit, [1], [1, 2], 0)
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.legfit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.legfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.legfit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.legfit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legfit, [1], [1], [])

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial.legfit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef3), y)
    coef3 = beignet.polynomial.legfit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef3), y)

    coef4 = beignet.polynomial.legfit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef4), y)
    coef4 = beignet.polynomial.legfit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef4), y)

    coef4 = beignet.polynomial.legfit(x, y, [2, 3, 4, 1, 0])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef4), y)

    coef2d = beignet.polynomial.legfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.legfit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial.legfit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial.legfit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial.legfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.legfit(x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(beignet.polynomial.legfit(x, x, 1), [0, 1])
    numpy.testing.assert_almost_equal(beignet.polynomial.legfit(x, x, [0, 1]), [0, 1])

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.polynomial.legfit(x, y, 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef1), y)
    coef2 = beignet.polynomial.legfit(x, y, [0, 2, 4])
    numpy.testing.assert_almost_equal(beignet.polynomial.legval(x, coef2), y)
    numpy.testing.assert_almost_equal(coef1, coef2)


def test_legfromroots():
    res = beignet.polynomial.legfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legtrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.legfromroots(roots)
        res = beignet.polynomial.legval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.leg2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(res, tgt)


def test_leggauss():
    x, w = beignet.polynomial.leggauss(100)

    v = beignet.polynomial.legvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))

    tgt = 2.0
    numpy.testing.assert_almost_equal(w.sum(), tgt)


def test_leggrid2d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.polynomial.leggrid2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.leggrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)


def test_leggrid3d():
    c1d = numpy.array([2.0, 2.0, 2.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.polynomial.leggrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.leggrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)


def test_legint():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legint, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], -1)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], 1, [0, 0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], lbnd=[0])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.legint, [0], scl=[0])
    numpy.testing.assert_raises(TypeError, beignet.polynomial.legint, [0], axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legint([0], m=i, k=k), [0, 1]
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        legpol = beignet.polynomial.poly2leg(pol)
        legint = beignet.polynomial.legint(legpol, m=1, k=[i])
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legtrim(
                beignet.polynomial.leg2poly(legint), tolerance=1e-6
            ),
            beignet.polynomial.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        legpol = beignet.polynomial.poly2leg(pol)
        legint = beignet.polynomial.legint(legpol, m=1, k=[i], lbnd=-1)
        numpy.testing.assert_almost_equal(beignet.polynomial.legval(-1, legint), i)

    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        legpol = beignet.polynomial.poly2leg(pol)
        legint = beignet.polynomial.legint(legpol, m=1, k=[i], scl=2)
        res = beignet.polynomial.leg2poly(legint)
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legtrim(res, tolerance=1e-6),
            beignet.polynomial.legtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for _ in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1)
            res = beignet.polynomial.legint(pol, m=j)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1, k=[k])
            res = beignet.polynomial.legint(pol, m=j, k=list(range(j)))
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1, k=[k], lbnd=-1)
            res = beignet.polynomial.legint(pol, m=j, k=list(range(j)), lbnd=-1)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = beignet.polynomial.legint(tgt, m=1, k=[k], scl=2)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.legtrim(
                    beignet.polynomial.legint(pol, m=j, k=list(range(j)), scl=2),
                    tolerance=1e-6,
                ),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legint(c2d, axis=0),
        numpy.vstack([beignet.polynomial.legint(c) for c in c2d.T]).T,
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legint(c2d, axis=1),
        numpy.vstack([beignet.polynomial.legint(c) for c in c2d]),
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legint(c2d, k=3, axis=1),
        numpy.vstack([beignet.polynomial.legint(c, k=3) for c in c2d]),
    )
    torch.testing.assert_close(beignet.polynomial.legint((1, 2, 3), 0), (1, 2, 3))


def test_legline():
    torch.testing.assert_close(beignet.polynomial.legline(3, 4), [3, 4])
    torch.testing.assert_close(beignet.polynomial.legline(3, 0), [3])
