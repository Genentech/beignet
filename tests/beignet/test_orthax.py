import functools
import math

import beignet.orthax
import jax
import numpy
import pytest
from jax import Array
from jax.numpy import (
    arange,
    array,
    cos,
    dot,
    einsum,
    empty,
    exp,
    eye,
    linspace,
    ones,
    sqrt,
    vstack,
    zeros,
    zeros_like,
)
from numpy.testing import (
    assert_array_almost_equal,
)

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(0)

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
    (array([1])),
    (array([0, 2])),
    (array([-2, 0, 4])),
    (array([0, -12, 0, 8])),
    (array([12, 0, -48, 0, 16])),
    (array([0, 120, 0, -160, 0, 32])),
    (array([-120, 0, 720, 0, -480, 0, 64])),
    (array([0, -1680, 0, 3360, 0, -1344, 0, 128])),
    (array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])),
    (array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])),
]

hermecoefficients = [
    (array([1])),
    (array([0, 1])),
    (array(array([-1, 0, 1]))),
    (array([0, -3, 0, 1])),
    (array([3, 0, -6, 0, 1])),
    (array([0, 15, 0, -10, 0, 1])),
    (array([-15, 0, 45, 0, -15, 0, 1])),
    (array([0, -105, 0, 105, 0, -21, 0, 1])),
    (array([105, 0, -420, 0, 210, 0, -28, 0, 1])),
    (array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])),
]

lagcoefficients = [
    (array([1]) / 1),
    (array([1, -1]) / 1),
    (array([2, -4, 1]) / 2),
    (array([6, -18, 9, -1]) / 6),
    (array([24, -96, 72, -16, 1]) / 24),
    (array([120, -600, 600, -200, 25, -1]) / 120),
    (array([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]

legcoefficients = [
    (array([1])),
    (array([0, 1])),
    (array([-1, 0, 3]) / 2),
    (array([0, -3, 0, 5]) / 2),
    (array([3, 0, -30, 0, 35]) / 8),
    (array([0, 15, 0, -70, 0, 63]) / 8),
    (array([-5, 0, 105, 0, -315, 0, 231]) / 16),
    (array([0, -35, 0, 315, 0, -693, 0, 429]) / 16),
    (array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128),
    (array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128),
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


def test__c_series_to_z_series():
    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax._c_series_to_z_series(
                array(
                    [2] + [1] * i,
                    numpy.double,
                ),
            ),
            array(
                [0.5] * i + [2] + [0.5] * i,
                numpy.double,
            ),
        )


def test__map_domain():
    assert_array_almost_equal(
        beignet.orthax._map_domain(
            [0, 4],
            [0, 4],
            [1, 3],
        ),
        [1, 3],
    )

    assert_array_almost_equal(
        beignet.orthax._map_domain(
            [0 - 1j, 2 + 1j],
            [0 - 1j, 2 + 1j],
            [-2, 2],
        ),
        [-2, 2],
    )

    assert_array_almost_equal(
        beignet.orthax._map_domain(
            array([[0, 4], [0, 4]]),
            [0, 4],
            [1, 3],
        ),
        array([[1, 3], [1, 3]]),
    )


def test__map_parameters():
    assert_array_almost_equal(
        beignet.orthax._map_parameters(
            [0, 4],
            [1, 3],
        ),
        [1, 0.5],
    )

    assert_array_almost_equal(
        beignet.orthax._map_parameters(
            [+0 - 1j, +2 + 1j],
            [-2 + 0j, +2 + 0j],
        ),
        [-1 + 1j, +1 - 1j],
    )


def test__pow():
    pytest.raises(
        ValueError,
        beignet.orthax._pow,
        (),
        array([1, 2, 3]),
        5,
        4,
    )


def test__trim_coefficients():
    pytest.raises(
        ValueError,
        beignet.orthax._trim_coefficients,
        array([2, -1, 1, 0]),
        -1,
    )

    assert_array_almost_equal(
        beignet.orthax._trim_coefficients(
            array([2, -1, 1, 0]),
        ),
        array([2, -1, 1, 0])[:-1],
    )

    assert_array_almost_equal(
        beignet.orthax._trim_coefficients(
            array([2, -1, 1, 0]),
            1,
        ),
        array([2, -1, 1, 0])[:-3],
    )

    assert_array_almost_equal(
        beignet.orthax._trim_coefficients(
            array(
                array([2, -1, 1, 0]),
            ),
            2,
        ),
        array([0]),
    )


def test__trim_sequence():
    for _ in range(5):
        assert_array_almost_equal(
            beignet.orthax._trim_sequence(array([1] + [0] * 5)),
            array([1]),
        )


def test__vandermonde():
    pytest.raises(
        ValueError,
        beignet.orthax._vander_nd,
        (),
        (1, 2, 3),
        [90],
    )

    pytest.raises(
        ValueError,
        beignet.orthax._vander_nd,
        (),
        (),
        [90.65],
    )

    pytest.raises(
        ValueError,
        beignet.orthax._vander_nd,
        (),
        (),
        array([]),
    )


def test__z_series_to_c_series():
    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax._z_series_to_c_series(
                array(
                    [0.5] * i + [2] + [0.5] * i,
                    numpy.float64,
                ),
            ),
            array(
                [2] + [1] * i,
                numpy.float64,
            ),
        )


def test_cheb2poly():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.cheb2poly(
                array([0] * i + [1]),
            ),
            chebcoefficients[i],
        )


def test_chebadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebcompanion():
    with pytest.raises(ValueError):
        beignet.orthax.chebcompanion(array([]))

    with pytest.raises(ValueError):
        beignet.orthax.chebcompanion(array([1]))

    for i in range(1, 5):
        assert beignet.orthax.chebcompanion(array([0] * i + [1])).shape == (i, i)

    assert beignet.orthax.chebcompanion(array([1, 2]))[0, 0] == -0.5


def test_chebder():
    with pytest.raises(TypeError):
        beignet.orthax.chebder(array([0]), 0.5)

    with pytest.raises(ValueError):
        beignet.orthax.chebder(array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.chebder(
                    array([0] * i + [1]),
                    order=0,
                ),
                tol=0.000001,
            ),
            beignet.orthax.chebtrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebder(
                        beignet.orthax.chebint(
                            array(
                                [0] * i + [1],
                            ),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(
                    array(
                        [0] * i + [1],
                    ),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebder(
                        beignet.orthax.chebint(
                            array([0] * i + [1]),
                            order=j,
                            scl=2,
                        ),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        beignet.orthax.chebder(c2d, axis=0),
        vstack([beignet.orthax.chebder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.chebder(
            c2d,
            axis=1,
        ),
        vstack([beignet.orthax.chebder(c) for c in c2d]),
    )


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.orthax.chebdiv(
                beignet.orthax.chebadd(
                    array([0] * i + [1]),
                    array([0] * j + [1]),
                ),
                array([0] * i + [1]),
            )

            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebadd(
                        beignet.orthax.chebmul(
                            quotient,
                            array([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(
                    beignet.orthax.chebadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebdomain():
    assert_array_almost_equal(
        beignet.orthax.chebdomain,
        array([-1, 1]),
    )


def test_chebfit():
    def f(x: Array) -> Array:
        return x * (x - 1) * (x - 2)

    def g(x: Array) -> Array:
        return x**4 + x**2 + 1

    with pytest.raises(ValueError):
        beignet.orthax.chebfit(
            array([1]),
            array([1]),
            degree=-1,
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([[1]]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([1]),
            array([[[1]]]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([1, 2]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([1]),
            array([1, 2]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([1]),
            array([1]),
            degree=0,
            weight=[[1]],
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([1]),
            array([1]),
            degree=0,
            weight=array([1, 1]),
        )

    with pytest.raises(ValueError):
        beignet.orthax.chebfit(
            array([1]),
            array([1]),
            degree=(-1,),
        )

    with pytest.raises(ValueError):
        beignet.orthax.chebfit(
            array([1]),
            array([1]),
            degree=(2, -1, 6),
        )

    with pytest.raises(TypeError):
        beignet.orthax.chebfit(
            array([1]),
            array([1]),
            degree=(),
        )

    x = linspace(0, 2, 50)
    y = f(x)

    coef3 = beignet.orthax.chebfit(
        x,
        y,
        degree=3,
    )

    assert_array_almost_equal(
        len(coef3),
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.chebval(
            x,
            coef3,
        ),
        y,
    )

    coef3 = beignet.orthax.chebfit(
        x,
        y,
        degree=(0, 1, 2, 3),
    )

    assert_array_almost_equal(
        len(coef3),
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.chebval(x, coef3),
        y,
    )

    coef4 = beignet.orthax.chebfit(
        x,
        y,
        degree=4,
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )
    assert_array_almost_equal(
        beignet.orthax.chebval(x, coef4),
        y,
    )
    coef4 = beignet.orthax.chebfit(
        x,
        y,
        degree=(0, 1, 2, 3, 4),
    )
    assert_array_almost_equal(
        len(coef4),
        5,
    )
    assert_array_almost_equal(
        beignet.orthax.chebval(x, coef4),
        y,
    )

    coef4 = beignet.orthax.chebfit(
        x,
        y,
        degree=(2, 3, 4, 1, 0),
    )
    assert_array_almost_equal(
        len(coef4),
        5,
    )
    assert_array_almost_equal(
        beignet.orthax.chebval(x, coef4),
        y,
    )

    coef2d = beignet.orthax.chebfit(
        x,
        array([y, y]).T,
        degree=3,
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    coef2d = beignet.orthax.chebfit(
        x,
        array([y, y]).T,
        degree=(0, 1, 2, 3),
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    w = zeros_like(x)
    yw = y.copy()

    w = w.at[1::2].set(1)

    wcoef3 = beignet.orthax.chebfit(
        x,
        yw,
        degree=3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef3 = beignet.orthax.chebfit(
        x,
        yw,
        degree=(0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef2d = beignet.orthax.chebfit(
        x,
        array([yw, yw]).T,
        degree=3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    wcoef2d = beignet.orthax.chebfit(
        x,
        array([yw, yw]).T,
        degree=(0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array(
            [
                coef3,
                coef3,
            ],
        ).T,
    )

    assert_array_almost_equal(
        beignet.orthax.chebfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=1,
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        beignet.orthax.chebfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        array([0, 1]),
    )

    input = linspace(-1, 1, 50)

    assert_array_almost_equal(
        beignet.orthax.chebval(
            input,
            beignet.orthax.chebfit(
                input,
                g(input),
                degree=4,
            ),
        ),
        g(input),
    )

    assert_array_almost_equal(
        beignet.orthax.chebval(
            input,
            beignet.orthax.chebfit(
                input,
                g(input),
                (0, 2, 4),
            ),
        ),
        g(input),
    )

    assert_array_almost_equal(
        beignet.orthax.chebfit(
            input,
            g(input),
            degree=4,
        ),
        beignet.orthax.chebfit(
            input,
            g(input),
            (0, 2, 4),
        ),
    )


def test_chebfromroots():
    assert_array_almost_equal(
        beignet.orthax.chebtrim(
            beignet.orthax.chebfromroots(
                array([]),
            ),
            tol=0.000001,
        ),
        array([1]),
    )
    for i in range(1, 5):
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.chebfromroots(
                    cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
                )
                * 2 ** (i - 1),
                tol=0.000001,
            ),
            beignet.orthax.chebtrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )


def test_chebgauss():
    x, w = beignet.orthax.chebgauss(100)

    v = beignet.orthax.chebvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_array_almost_equal(vv, eye(100))
    assert_array_almost_equal(w.sum(), math.pi)


def test_chebgrid2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.chebgrid2d(
            x1,
            x2,
            einsum("i,j->ij", array([2.5, 2.0, 1.5]), array([2.5, 2.0, 1.5])),
        ),
        einsum("i,j->ij", y1, y2),
    )

    z = ones([2, 3])
    res = beignet.orthax.chebgrid2d(
        z,
        z,
        einsum("i,j->ij", array([2.5, 2.0, 1.5]), array([2.5, 2.0, 1.5])),
    )
    assert res.shape == (2, 3) * 2


def test_chebgrid3d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.chebgrid3d(
            x1,
            x2,
            x3,
            einsum(
                "i,j,k->ijk",
                array([2.5, 2.0, 1.5]),
                array([2.5, 2.0, 1.5]),
                array([2.5, 2.0, 1.5]),
            ),
        ),
        einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    z = ones([2, 3])
    res = beignet.orthax.chebgrid3d(
        z,
        z,
        z,
        einsum(
            "i,j,k->ijk",
            array([2.5, 2.0, 1.5]),
            array([2.5, 2.0, 1.5]),
            array([2.5, 2.0, 1.5]),
        ),
    )
    assert res.shape == (2, 3) * 3


def test_chebint():
    pytest.raises(TypeError, beignet.orthax.chebint, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.chebint, array([0]), -1)
    pytest.raises(ValueError, beignet.orthax.chebint, array([0]), 1, [0, 0])
    pytest.raises(ValueError, beignet.orthax.chebint, array([0]), lbnd=[0])
    pytest.raises(ValueError, beignet.orthax.chebint, array([0]), scl=[0])
    pytest.raises(TypeError, beignet.orthax.chebint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.chebint(array([0]), order=i, k=k), tol=0.000001
            ),
            [0, 1],
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.cheb2poly(
                    beignet.orthax.chebint(
                        beignet.orthax.poly2cheb(array([0] * i + [1])), order=1, k=[i]
                    )
                ),
                tol=0.000001,
            ),
            beignet.orthax.chebtrim(array([i] + [0] * i + [1 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.chebval(
                -1,
                beignet.orthax.chebint(
                    beignet.orthax.poly2cheb(array([0] * i + [1])),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.cheb2poly(
                    beignet.orthax.chebint(
                        beignet.orthax.poly2cheb(array([0] * i + [1])),
                        order=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tol=0.000001,
            ),
            beignet.orthax.chebtrim(array([i] + [0] * i + [2 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = beignet.orthax.chebint(target, order=1)
            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebint(pol, order=j), tol=0.000001
                ),
                beignet.orthax.chebtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]

            for k in range(j):
                target = beignet.orthax.chebint(target, order=1, k=[k])

            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebint(pol, order=j, k=list(range(j))), tol=0.000001
                ),
                beignet.orthax.chebtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.chebint(target, order=1, k=[k], lbnd=-1)
            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebint(pol, order=j, k=list(range(j)), lbnd=-1),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.chebint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([beignet.orthax.chebint(c) for c in c2d.T]).T
    assert_array_almost_equal(
        beignet.orthax.chebint(c2d, axis=0),
        target,
    )

    target = vstack([beignet.orthax.chebint(c) for c in c2d])
    assert_array_almost_equal(
        beignet.orthax.chebint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = vstack([beignet.orthax.chebint(c, k=3) for c in c2d])
    assert_array_almost_equal(
        beignet.orthax.chebint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )


def test_chebinterpolate():
    def f(x):
        return x * (x - 1) * (x - 2)

    pytest.raises(ValueError, beignet.orthax.chebinterpolate, f, -1)

    for deg in range(1, 5):
        assert beignet.orthax.chebinterpolate(f, deg).shape == (deg + 1,)

    def powx(x, p):
        return x**p

    x = linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = beignet.orthax.chebinterpolate(powx, deg, (p,))
            assert_array_almost_equal(
                beignet.orthax.chebval(x, c),
                powx(x, p),
            )


def test_chebline():
    assert_array_almost_equal(beignet.orthax.chebline(3, 4), array([3, 4]))


def test_chebmul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[abs(i + j)].set(target[abs(i + j)] + 0.5)
            target = target.at[abs(i - j)].set(target[abs(i - j)] + 0.5)

            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebmul(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebmulx():
    assert_array_almost_equal(
        beignet.orthax.chebtrim(beignet.orthax.chebmulx([0]), tol=0.000001), [0]
    )

    assert_array_almost_equal(
        beignet.orthax.chebtrim(beignet.orthax.chebmulx([1]), tol=0.000001), [0, 1]
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.chebmulx(array([0] * i + [1])), tol=0.000001
            ),
            [0] * (i - 1) + [0.5, 0, 0.5],
        )


def test_chebone():
    assert_array_almost_equal(
        beignet.orthax.chebone,
        array([1]),
    )


def test_chebpow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebpow(arange(i + 1), j), tol=0.000001
                ),
                beignet.orthax.chebtrim(
                    functools.reduce(
                        beignet.orthax.chebmul,
                        [(arange(i + 1))] * j,
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebpts1():
    with pytest.raises(ValueError):
        beignet.orthax.chebpts1(1.5)

    with pytest.raises(ValueError):
        beignet.orthax.chebpts1(0)

    assert_array_almost_equal(beignet.orthax.chebpts1(1), [0])

    assert_array_almost_equal(
        beignet.orthax.chebpts1(2), [-0.70710678118654746, 0.70710678118654746]
    )

    assert_array_almost_equal(
        beignet.orthax.chebpts1(3), [-0.86602540378443871, 0, 0.86602540378443871]
    )

    assert_array_almost_equal(
        beignet.orthax.chebpts1(4),
        [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325],
    )


def test_chebpts2():
    with pytest.raises(ValueError):
        beignet.orthax.chebpts2(1.5)

    with pytest.raises(ValueError):
        beignet.orthax.chebpts2(1)

    assert_array_almost_equal(beignet.orthax.chebpts2(2), [-1, 1])

    assert_array_almost_equal(beignet.orthax.chebpts2(3), array([-1, 0, 1]))

    assert_array_almost_equal(beignet.orthax.chebpts2(4), [-1, -0.5, 0.5, 1])

    assert_array_almost_equal(
        beignet.orthax.chebpts2(5), [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
    )


def test_chebroots():
    assert_array_almost_equal(
        beignet.orthax.chebroots([1]),
        array([]),
    )

    assert_array_almost_equal(
        beignet.orthax.chebroots(array([1, 2])),
        array([-0.5]),
    )

    for i in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.chebtrim(
                beignet.orthax.chebroots(
                    beignet.orthax.chebfromroots(
                        linspace(-1, 1, i),
                    )
                ),
                tol=0.000001,
            ),
            beignet.orthax.chebtrim(
                linspace(-1, 1, i),
                tol=0.000001,
            ),
        )


def test_chebsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                beignet.orthax.chebtrim(
                    beignet.orthax.chebsub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebtrim():
    pytest.raises(ValueError, beignet.orthax.chebtrim, array([2, -1, 1, 0]), -1)

    assert_array_almost_equal(
        beignet.orthax.chebtrim(array([2, -1, 1, 0])), array([2, -1, 1, 0])[:-1]
    )

    assert_array_almost_equal(
        beignet.orthax.chebtrim(array([2, -1, 1, 0]), 1), array([2, -1, 1, 0])[:-3]
    )

    assert_array_almost_equal(
        beignet.orthax.chebtrim(array([2, -1, 1, 0]), 2),
        array([0]),
    )


def test_chebval():
    assert_array_almost_equal(beignet.orthax.chebval(array([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [beignet.orthax.polyval(x, c) for c in chebcoefficients]
    for i in range(10):
        target = y[i]
        res = beignet.orthax.chebval(x, array([0] * i + [1]))
        assert_array_almost_equal(
            res,
            target,
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(beignet.orthax.chebval(x, [1]).shape, dims)
        assert_array_almost_equal(beignet.orthax.chebval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(
            beignet.orthax.chebval(x, array([1, 0, 0])).shape, dims
        )


def test_chebval2d():
    c1d = array([2.5, 2.0, 1.5])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        beignet.orthax.chebval2d,
        x1,
        x2[:2],
        c2d,
    )

    target = y1 * y2
    res = beignet.orthax.chebval2d(
        x1,
        x2,
        c2d,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = beignet.orthax.chebval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_chebval3d():
    c1d = array([2.5, 2.0, 1.5])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, beignet.orthax.chebval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    res = beignet.orthax.chebval3d(x1, x2, x3, c3d)
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = beignet.orthax.chebval3d(z, z, z, c3d)
    assert res.shape == (2, 3)


def test_chebvander():
    x = arange(3)
    v = beignet.orthax.chebvander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], beignet.orthax.chebval(x, coef))

    x = array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.chebvander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], beignet.orthax.chebval(x, coef))


def test_chebvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    van = beignet.orthax.chebvander2d(x1, x2, (1, 2))
    target = beignet.orthax.chebval2d(x1, x2, c)
    res = dot(van, c.ravel())
    assert_array_almost_equal(
        res,
        target,
    )

    van = beignet.orthax.chebvander2d([x1], [x2], (1, 2))
    assert van.shape == (1, 5, 6)


def test_chebvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(beignet.orthax.chebvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        beignet.orthax.chebval3d(x1, x2, x3, c),
    )

    van = beignet.orthax.chebvander3d([x1], [x2], [x3], (1, 2, 3))
    assert van.shape == (1, 5, 24)


def test_chebweight():
    x = linspace(-1, 1, 11)[1:-1]
    assert_array_almost_equal(
        beignet.orthax.chebweight(x),
        1.0 / (sqrt(1 + x) * sqrt(1 - x)),
    )


def test_chebx():
    assert_array_almost_equal(
        beignet.orthax.chebx,
        array([0, 1]),
    )


def test_chebzero():
    assert_array_almost_equal(beignet.orthax.chebzero, [0])


def test_getdomain():
    assert_array_almost_equal(
        beignet.orthax.getdomain([1, 10, 3, -1]),
        [-1, 10],
    )

    assert_array_almost_equal(
        beignet.orthax.getdomain([1 + 1j, 1 - 1j, 0, 2]),
        [-1j, 2 + 1j],
    )


def test_herm2poly():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.herm2poly(array([0] * i + [1])),
            hermcoefficients[i],
        )


def test_hermadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermcompanion():
    with pytest.raises(ValueError):
        beignet.orthax.hermcompanion(array([]))
    with pytest.raises(ValueError):
        beignet.orthax.hermcompanion([1])

    for i in range(1, 5):
        assert beignet.orthax.hermcompanion(array([0] * i + [1])).shape == (i, i)

    assert beignet.orthax.hermcompanion(array([1, 2]))[0, 0] == -0.25


def test_hermder():
    with pytest.raises(TypeError):
        beignet.orthax.hermder(array([0]), 0.5)

    with pytest.raises(ValueError):
        beignet.orthax.hermder(array([0]), -1)

    for i in range(5):
        target = array([0] * i + [1])
        res = beignet.orthax.hermder(target, order=0)
        assert_array_almost_equal(
            beignet.orthax.hermtrim(res, tol=0.000001),
            beignet.orthax.hermtrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            target = array([0] * i + [1])
            res = beignet.orthax.hermder(
                beignet.orthax.hermint(target, order=j), order=j
            )
            assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=0.000001),
                beignet.orthax.hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            target = array([0] * i + [1])
            res = beignet.orthax.hermder(
                beignet.orthax.hermint(target, order=j, scl=2), order=j, scl=0.5
            )
            assert_array_almost_equal(
                beignet.orthax.hermtrim(res, tol=0.000001),
                beignet.orthax.hermtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([beignet.orthax.hermder(c) for c in c2d.T]).T
    res = beignet.orthax.hermder(c2d, axis=0)
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([beignet.orthax.hermder(c) for c in c2d])
    res = beignet.orthax.hermder(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )


def test_hermdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.orthax.hermdiv(
                beignet.orthax.hermadd(
                    array([0.0] * i + [1.0]),
                    array([0.0] * j + [1.0]),
                ),
                array([0.0] * i + [1.0]),
            )

            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermadd(
                        beignet.orthax.hermmul(
                            quotient,
                            array([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermtrim(
                    beignet.orthax.hermadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermdomain():
    assert_array_almost_equal(beignet.orthax.hermdomain, array([-1, 1]))


def test_herme2poly():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.herme2poly(array([0] * i + [1])),
            hermecoefficients[i],
        )


def test_hermeadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                beignet.orthax.hermetrim(
                    beignet.orthax.hermeadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermecompanion():
    with pytest.raises(ValueError):
        beignet.orthax.hermecompanion(array([]))

    with pytest.raises(ValueError):
        beignet.orthax.hermecompanion([1])

    for i in range(1, 5):
        coef = array([0] * i + [1])
        assert beignet.orthax.hermecompanion(coef).shape == (i, i)

    assert beignet.orthax.hermecompanion(array([1, 2]))[0, 0] == -0.5


def test_hermeder():
    pytest.raises(TypeError, beignet.orthax.hermeder, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.hermeder, array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.hermetrim(
                beignet.orthax.hermeder(array([0] * i + [1]), order=0), tol=0.000001
            ),
            beignet.orthax.hermetrim(array([0] * i + [1]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.hermetrim(
                    beignet.orthax.hermeder(
                        beignet.orthax.hermeint(array([0] * i + [1]), order=j),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermetrim(array([0] * i + [1]), tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.hermetrim(
                    beignet.orthax.hermeder(
                        beignet.orthax.hermeint(array([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermetrim(array([0] * i + [1]), tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        beignet.orthax.hermeder(c2d, axis=0),
        vstack([beignet.orthax.hermeder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeder(
            c2d,
            axis=1,
        ),
        vstack([beignet.orthax.hermeder(c) for c in c2d]),
    )


def test_hermediv():
    for i in range(5):
        for j in range(5):
            ci = array([0] * i + [1])
            cj = array([0] * j + [1])
            quotient, remainder = beignet.orthax.hermediv(
                beignet.orthax.hermeadd(ci, cj), ci
            )
            assert_array_almost_equal(
                beignet.orthax.hermetrim(
                    beignet.orthax.hermeadd(
                        beignet.orthax.hermemul(quotient, ci), remainder
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermetrim(beignet.orthax.hermeadd(ci, cj), tol=0.000001),
            )


def test_hermedomain():
    assert_array_almost_equal(beignet.orthax.hermedomain, [-1, 1])


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    with pytest.raises(ValueError):
        beignet.orthax.hermefit(
            array([1]),
            array([1]),
            degree=-1,
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([[1]]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([1]),
            array([[[1]]]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([1, 2]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([1]),
            array([1, 2]),
            degree=0,
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([1]),
            array([1]),
            degree=0,
            weight=[[1]],
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([1]),
            array([1]),
            degree=0,
            weight=array([1, 1]),
        )

    with pytest.raises(ValueError):
        beignet.orthax.hermefit(
            array([1]),
            array([1]),
            degree=(-1,),
        )

    with pytest.raises(ValueError):
        beignet.orthax.hermefit(
            array([1]),
            array([1]),
            degree=(2, -1, 6),
        )

    with pytest.raises(TypeError):
        beignet.orthax.hermefit(
            array([1]),
            array([1]),
            degree=(),
        )

    x = linspace(0, 2)

    y = f(x)

    coef3 = beignet.orthax.hermefit(
        x,
        y,
        3,
    )

    assert_array_almost_equal(
        len(coef3),
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, coef3),
        y,
    )

    coef3 = beignet.orthax.hermefit(
        x,
        y,
        (0, 1, 2, 3),
    )

    assert_array_almost_equal(
        len(coef3),
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, coef3),
        y,
    )

    coef4 = beignet.orthax.hermefit(
        x,
        y,
        4,
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, coef4),
        y,
    )

    coef4 = beignet.orthax.hermefit(
        x,
        y,
        (0, 1, 2, 3, 4),
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, coef4),
        y,
    )

    coef4 = beignet.orthax.hermefit(
        x,
        y,
        (2, 3, 4, 1, 0),
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, coef4),
        y,
    )

    coef2d = beignet.orthax.hermefit(
        x,
        array([y, y]).T,
        3,
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    coef2d = beignet.orthax.hermefit(
        x,
        array([y, y]).T,
        (0, 1, 2, 3),
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    w = zeros_like(x)

    yw = y.copy()

    w = w.at[1::2].set(1)
    y = y.at[0::2].set(0)

    wcoef3 = beignet.orthax.hermefit(
        x,
        yw,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef3 = beignet.orthax.hermefit(
        x,
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef2d = beignet.orthax.hermefit(
        x,
        array([yw, yw]).T,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    wcoef2d = beignet.orthax.hermefit(
        x,
        array([yw, yw]).T,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    x = array([1, 1j, -1, -1j])

    assert_array_almost_equal(
        beignet.orthax.hermefit(x, x, 1),
        array([0, 1]),
    )

    assert_array_almost_equal(
        beignet.orthax.hermefit(x, x, (0, 1)),
        array([0, 1]),
    )

    x = linspace(-1, 1)

    y = f2(x)

    coef1 = beignet.orthax.hermefit(x, y, 4)

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, coef1),
        y,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval(x, beignet.orthax.hermefit(x, y, (0, 2, 4))),
        y,
    )

    assert_array_almost_equal(
        coef1,
        beignet.orthax.hermefit(x, y, (0, 2, 4)),
    )


def test_hermefromroots():
    res = beignet.orthax.hermefromroots(array([]))
    assert_array_almost_equal(
        beignet.orthax.hermetrim(res, tol=0.000001),
        array([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.hermefromroots(roots)
        assert len(pol) == i + 1
        assert_array_almost_equal(beignet.orthax.herme2poly(pol)[-1], 1)
        assert_array_almost_equal(
            beignet.orthax.hermeval(roots, pol),
            0,
        )


def test_hermegauss():
    x, w = beignet.orthax.hermegauss(100)

    v = beignet.orthax.hermevander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_array_almost_equal(vv, eye(100))

    target = sqrt(2 * math.pi)
    assert_array_almost_equal(w.sum(), target)


def test_hermegrid2d():
    c1d = array([4.0, 2.0, 3.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j->ij", y1, y2)
    res = beignet.orthax.hermegrid2d(
        x1,
        x2,
        c2d,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = beignet.orthax.hermegrid2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3) * 2


def test_hermegrid3d():
    c1d = array([4.0, 2.0, 3.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    assert_array_almost_equal(
        beignet.orthax.hermegrid3d(x1, x2, x3, c3d),
        einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = ones([2, 3])
    res = beignet.orthax.hermegrid3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3


def test_hermeint():
    pytest.raises(TypeError, beignet.orthax.hermeint, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.hermeint, array([0]), -1)
    pytest.raises(ValueError, beignet.orthax.hermeint, array([0]), 1, [0, 0])
    pytest.raises(ValueError, beignet.orthax.hermeint, array([0]), lbnd=[0])
    pytest.raises(ValueError, beignet.orthax.hermeint, array([0]), scl=[0])
    pytest.raises(TypeError, beignet.orthax.hermeint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = beignet.orthax.hermeint(array([0]), order=i, k=k)
        assert_array_almost_equal(
            beignet.orthax.hermetrim(res, tol=0.000001),
            array([0, 1]),
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [1 / scl]
        hermepol = beignet.orthax.poly2herme(pol)
        hermeint = beignet.orthax.hermeint(hermepol, order=1, k=[i])
        res = beignet.orthax.herme2poly(hermeint)
        assert_array_almost_equal(
            beignet.orthax.hermetrim(res, tol=0.000001),
            beignet.orthax.hermetrim(target, tol=0.000001),
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        hermepol = beignet.orthax.poly2herme(pol)
        hermeint = beignet.orthax.hermeint(hermepol, order=1, k=[i], lbnd=-1)
        assert_array_almost_equal(beignet.orthax.hermeval(-1, hermeint), i)

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [2 / scl]
        hermepol = beignet.orthax.poly2herme(pol)
        hermeint = beignet.orthax.hermeint(hermepol, order=1, k=[i], scl=2)
        res = beignet.orthax.herme2poly(hermeint)
        assert_array_almost_equal(
            beignet.orthax.hermetrim(res, tol=0.000001),
            beignet.orthax.hermetrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = beignet.orthax.hermeint(target, order=1)
            res = beignet.orthax.hermeint(pol, order=j)
            assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=0.000001),
                beignet.orthax.hermetrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.hermeint(target, order=1, k=[k])
            res = beignet.orthax.hermeint(pol, order=j, k=list(range(j)))
            assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=0.000001),
                beignet.orthax.hermetrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.hermeint(target, order=1, k=[k], lbnd=-1)
            res = beignet.orthax.hermeint(pol, order=j, k=list(range(j)), lbnd=-1)
            assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=0.000001),
                beignet.orthax.hermetrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.hermeint(target, order=1, k=[k], scl=2)
            res = beignet.orthax.hermeint(pol, order=j, k=list(range(j)), scl=2)
            assert_array_almost_equal(
                beignet.orthax.hermetrim(res, tol=0.000001),
                beignet.orthax.hermetrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([beignet.orthax.hermeint(c) for c in c2d.T]).T
    res = beignet.orthax.hermeint(c2d, axis=0)
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([beignet.orthax.hermeint(c) for c in c2d])
    res = beignet.orthax.hermeint(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([beignet.orthax.hermeint(c, k=3) for c in c2d])
    res = beignet.orthax.hermeint(
        c2d,
        k=3,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )


def test_hermeline():
    assert_array_almost_equal(beignet.orthax.hermeline(3, 4), array([3, 4]))


def test_hermemul():
    x = linspace(-3, 3, 100)
    for i in range(5):
        pol1 = array([0] * i + [1])
        val1 = beignet.orthax.hermeval(x, pol1)
        for j in range(5):
            pol2 = array([0] * j + [1])
            val2 = beignet.orthax.hermeval(x, pol2)
            pol3 = beignet.orthax.hermemul(pol1, pol2)
            val3 = beignet.orthax.hermeval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_array_almost_equal(val3, val1 * val2, decimal=3)


def test_hermemulx():
    assert_array_almost_equal(
        beignet.orthax.hermetrim(beignet.orthax.hermemulx([0]), tol=0.000001), [0]
    )
    assert_array_almost_equal(
        beignet.orthax.hermetrim(beignet.orthax.hermemulx([1]), tol=0.000001), [0, 1]
    )
    for i in range(1, 5):
        assert_array_almost_equal(
            beignet.orthax.hermetrim(
                beignet.orthax.hermemulx(array([0] * i + [1])), tol=0.000001
            ),
            [0] * (i - 1) + [i, 0, 1],
        )


def test_hermeone():
    assert_array_almost_equal(
        beignet.orthax.hermeone,
        array([1]),
    )


def test_hermepow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)
            assert_array_almost_equal(
                beignet.orthax.hermetrim(beignet.orthax.hermepow(c, j), tol=0.000001),
                beignet.orthax.hermetrim(
                    functools.reduce(beignet.orthax.hermemul, [c] * j, array([1])),
                    tol=0.000001,
                ),
            )


def test_hermeroots():
    assert_array_almost_equal(beignet.orthax.hermeroots([1]), array([]))

    assert_array_almost_equal(beignet.orthax.hermeroots([1, 1]), [-1])

    for i in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.hermetrim(
                beignet.orthax.hermeroots(
                    beignet.orthax.hermefromroots(
                        linspace(-1, 1, i),
                    )
                ),
                tol=0.000001,
            ),
            beignet.orthax.hermetrim(linspace(-1, 1, i), tol=0.000001),
        )


def test_hermesub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                beignet.orthax.hermetrim(
                    beignet.orthax.hermesub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermetrim():
    coef = array([2, -1, 1, 0])

    pytest.raises(ValueError, beignet.orthax.hermetrim, coef, -1)

    assert_array_almost_equal(beignet.orthax.hermetrim(coef), coef[:-1])
    assert_array_almost_equal(beignet.orthax.hermetrim(coef, 1), coef[:-3])
    assert_array_almost_equal(beignet.orthax.hermetrim(coef, 2), [0])


def test_hermeval():
    assert_array_almost_equal(beignet.orthax.hermeval(array([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [beignet.orthax.polyval(x, c) for c in hermecoefficients]
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.hermeval(x, array([0] * i + [1])), y[i], decimal=4
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(beignet.orthax.hermeval(x, [1]).shape, dims)
        assert_array_almost_equal(beignet.orthax.hermeval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(
            beignet.orthax.hermeval(x, array([1, 0, 0])).shape, dims
        )


def test_hermeval2d():
    c1d = array([4.0, 2.0, 3.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    pytest.raises(
        ValueError,
        beignet.orthax.hermeval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_array_almost_equal(
        beignet.orthax.hermeval2d(
            x1,
            x2,
            c2d,
        ),
        y1 * y2,
    )

    z = ones([2, 3])
    res = beignet.orthax.hermeval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_hermeval3d():
    c1d = array([4.0, 2.0, 3.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, beignet.orthax.hermeval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    res = beignet.orthax.hermeval3d(x1, x2, x3, c3d)
    assert_array_almost_equal(res, target, decimal=4)

    z = ones([2, 3])
    res = beignet.orthax.hermeval3d(z, z, z, c3d)
    assert res.shape == (2, 3)


def test_hermevander():
    x = arange(3)
    v = beignet.orthax.hermevander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], beignet.orthax.hermeval(x, coef))

    x = array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.hermevander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], beignet.orthax.hermeval(x, coef))


def test_hermevander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(beignet.orthax.hermevander2d(x1, x2, (1, 2)), c.ravel()),
        beignet.orthax.hermeval2d(x1, x2, c),
    )

    van = beignet.orthax.hermevander2d([x1], [x2], (1, 2))
    assert van.shape == (1, 5, 6)


def test_hermevander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    van = beignet.orthax.hermevander3d(x1, x2, x3, (1, 2, 3))
    assert_array_almost_equal(
        dot(van, c.ravel()), beignet.orthax.hermeval3d(x1, x2, x3, c)
    )

    van = beignet.orthax.hermevander3d([x1], [x2], [x3], (1, 2, 3))
    assert van.shape == (1, 5, 24)


def test_hermeweight():
    x = linspace(-5, 5, 11)
    target = exp(-0.5 * x**2)
    res = beignet.orthax.hermeweight(x)
    assert_array_almost_equal(
        res,
        target,
    )


def test_hermex():
    assert_array_almost_equal(
        beignet.orthax.hermex,
        array([0, 1]),
    )


def test_hermezero():
    assert_array_almost_equal(
        beignet.orthax.hermezero,
        array([0]),
    )


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    pytest.raises(ValueError, beignet.orthax.hermfit, [1], [1], -1)
    pytest.raises(TypeError, beignet.orthax.hermfit, [[1]], [1], 0)
    pytest.raises(TypeError, beignet.orthax.hermfit, array([]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.hermfit, [1], [[[1]]], 0)
    pytest.raises(TypeError, beignet.orthax.hermfit, array([1, 2]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.hermfit, [1], array([1, 2]), 0)
    pytest.raises(TypeError, beignet.orthax.hermfit, [1], [1], 0, weight=[[1]])

    pytest.raises(TypeError, beignet.orthax.hermfit, [1], [1], 0, weight=[1, 1])

    pytest.raises(ValueError, beignet.orthax.hermfit, [1], [1], (-1))

    pytest.raises(ValueError, beignet.orthax.hermfit, [1], [1], (2, -1, 6))

    pytest.raises(TypeError, beignet.orthax.hermfit, [1], [1], ())

    x = linspace(0, 2)

    y = f(x)

    coef3 = beignet.orthax.hermfit(
        x,
        y,
        3,
    )

    assert len(coef3) == 4

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef3),
        y,
    )

    coef3 = beignet.orthax.hermfit(
        x,
        y,
        (0, 1, 2, 3),
    )

    assert len(coef3) == 4

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef3),
        y,
    )

    coef4 = beignet.orthax.hermfit(
        x,
        y,
        4,
    )

    assert len(coef4) == 5

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef4),
        y,
    )

    coef4 = beignet.orthax.hermfit(
        x,
        y,
        (0, 1, 2, 3, 4),
    )

    assert len(coef4) == 5

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef4),
        y,
    )

    coef4 = beignet.orthax.hermfit(
        x,
        y,
        (2, 3, 4, 1, 0),
    )

    assert len(coef4) == 5

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef4),
        y,
    )

    coef2d = beignet.orthax.hermfit(
        x,
        array([y, y]).T,
        3,
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    coef2d = beignet.orthax.hermfit(
        x,
        array([y, y]).T,
        (0, 1, 2, 3),
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    w = zeros_like(x)

    yw = y.copy()

    w = w.at[1::2].set(1)
    y = y.at[0::2].set(0)

    wcoef3 = beignet.orthax.hermfit(
        x,
        yw,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef3 = beignet.orthax.hermfit(
        x,
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef2d = beignet.orthax.hermfit(
        x,
        array([yw, yw]).T,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    wcoef2d = beignet.orthax.hermfit(
        x,
        array([yw, yw]).T,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    x = [1, 1j, -1, -1j]

    assert_array_almost_equal(
        beignet.orthax.hermfit(x, x, 1),
        [0, 0.5],
    )

    assert_array_almost_equal(
        beignet.orthax.hermfit(x, x, (0, 1)),
        [0, 0.5],
    )

    x = linspace(-1, 1)

    y = f2(x)

    coef1 = beignet.orthax.hermfit(
        x,
        y,
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef1),
        y,
    )

    coef2 = beignet.orthax.hermfit(
        x,
        y,
        (0, 2, 4),
    )

    assert_array_almost_equal(
        beignet.orthax.hermval(x, coef2),
        y,
    )

    assert_array_almost_equal(
        coef1,
        coef2,
    )


def test_hermfromroots():
    res = beignet.orthax.hermfromroots(array([]))
    assert_array_almost_equal(
        beignet.orthax.hermtrim(res, tol=0.000001),
        array([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.hermfromroots(roots)
        res = beignet.orthax.hermval(roots, pol)
        target = 0
        assert len(pol) == i + 1
        assert_array_almost_equal(beignet.orthax.herm2poly(pol)[-1], 1)
        assert_array_almost_equal(
            res,
            target,
        )


def test_hermgauss():
    x, w = beignet.orthax.hermgauss(100)

    v = beignet.orthax.hermvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_array_almost_equal(
        vv,
        eye(100),
    )

    target = sqrt(math.pi)
    assert_array_almost_equal(w.sum(), target)


def test_hermgrid2d():
    c1d = array([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    target = einsum("i,j->ij", y1, y2)
    assert_array_almost_equal(
        beignet.orthax.hermgrid2d(
            x1,
            x2,
            c2d,
        ),
        target,
    )

    z = ones([2, 3])
    assert (
        beignet.orthax.hermgrid2d(
            z,
            z,
            c2d,
        ).shape
        == (2, 3) * 2
    )


def test_hermgrid3d():
    c1d = array([2.5, 1.0, 0.75])
    c3d = einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.hermgrid3d(
            x1,
            x2,
            x3,
            c3d,
        ),
        einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    z = ones([2, 3])

    assert beignet.orthax.hermgrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_hermint():
    pytest.raises(TypeError, beignet.orthax.hermint, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.hermint, array([0]), -1)
    pytest.raises(
        ValueError,
        beignet.orthax.hermint,
        array([0]),
        1,
        array([0, 0]),
    )
    pytest.raises(ValueError, beignet.orthax.hermint, array([0]), lbnd=[0])
    pytest.raises(ValueError, beignet.orthax.hermint, array([0]), scl=[0])
    pytest.raises(TypeError, beignet.orthax.hermint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_array_almost_equal(
            beignet.orthax.hermtrim(
                beignet.orthax.hermint(array([0]), order=i, k=k), tol=0.000001
            ),
            [0, 0.5],
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        hermpol = beignet.orthax.poly2herm(pol)
        hermint = beignet.orthax.hermint(hermpol, order=1, k=[i])
        assert_array_almost_equal(
            beignet.orthax.hermtrim(beignet.orthax.herm2poly(hermint), tol=0.000001),
            beignet.orthax.hermtrim([i] + [0] * i + [1 / scl], tol=0.000001),
        )

    for i in range(5):
        pol = array([0] * i + [1])
        hermpol = beignet.orthax.poly2herm(pol)
        hermint = beignet.orthax.hermint(hermpol, order=1, k=[i], lbnd=-1)
        assert_array_almost_equal(beignet.orthax.hermval(-1, hermint), i)

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        hermpol = beignet.orthax.poly2herm(pol)
        hermint = beignet.orthax.hermint(hermpol, order=1, k=[i], scl=2)
        assert_array_almost_equal(
            beignet.orthax.hermtrim(beignet.orthax.herm2poly(hermint), tol=0.000001),
            beignet.orthax.hermtrim([i] + [0] * i + [2 / scl], tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = beignet.orthax.hermint(target, order=1)
            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermint(pol, order=j), tol=0.000001
                ),
                beignet.orthax.hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.hermint(target, order=1, k=[k])
            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermint(pol, order=j, k=list(range(j))), tol=0.000001
                ),
                beignet.orthax.hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.hermint(target, order=1, k=[k], lbnd=-1)
            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermint(pol, order=j, k=list(range(j)), lbnd=-1),
                    tol=0.000001,
                ),
                beignet.orthax.hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.hermint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                beignet.orthax.hermtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([beignet.orthax.hermint(c) for c in c2d.T]).T
    assert_array_almost_equal(beignet.orthax.hermint(c2d, axis=0), target)

    target = vstack([beignet.orthax.hermint(c) for c in c2d])
    assert_array_almost_equal(
        beignet.orthax.hermint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = vstack([beignet.orthax.hermint(c, k=3) for c in c2d])
    assert_array_almost_equal(
        beignet.orthax.hermint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )


def test_hermline():
    assert_array_almost_equal(beignet.orthax.hermline(3, 4), [3, 2])


def test_hermmul():
    x = linspace(-3, 3, 100)

    for i in range(5):
        pol1 = array([0.0] * i + [1.0])
        val1 = beignet.orthax.hermval(x, pol1)
        for j in range(5):
            pol2 = array([0.0] * j + [1.0])
            val2 = beignet.orthax.hermval(x, pol2)
            pol3 = beignet.orthax.hermmul(pol1, pol2)
            val3 = beignet.orthax.hermval(x, pol3)

            assert len(beignet.orthax.hermtrim(pol3, tol=0.000001)) == i + j + 1

            assert_array_almost_equal(
                val3,
                val1 * val2,
            )


def test_hermmulx():
    assert_array_almost_equal(
        beignet.orthax.hermtrim(beignet.orthax.hermmulx([0.0]), tol=0.000001), [0.0]
    )
    assert_array_almost_equal(beignet.orthax.hermmulx([1.0]), [0.0, 0.5])
    for i in range(1, 5):
        assert_array_almost_equal(
            beignet.orthax.hermmulx(array([0.0] * i + [1.0])),
            [0.0] * (i - 1) + [i, 0.0, 0.5],
        )


def test_hermone():
    assert_array_almost_equal(beignet.orthax.hermone, array([1]))


def test_hermpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1).astype(float)
            assert_array_almost_equal(
                beignet.orthax.hermtrim(beignet.orthax.hermpow(c, j), tol=0.000001),
                beignet.orthax.hermtrim(
                    functools.reduce(beignet.orthax.hermmul, [c] * j, array([1])),
                    tol=0.000001,
                ),
            )


def test_hermroots():
    assert_array_almost_equal(beignet.orthax.hermroots([1]), array([]))
    assert_array_almost_equal(beignet.orthax.hermroots([1, 1]), array([-0.5]))
    for i in range(2, 5):
        target = linspace(-1, 1, i)
        assert_array_almost_equal(
            beignet.orthax.hermtrim(
                beignet.orthax.hermroots(beignet.orthax.hermfromroots(target)),
                tol=0.000001,
            ),
            beignet.orthax.hermtrim(target, tol=0.000001),
        )


def test_hermsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                beignet.orthax.hermtrim(
                    beignet.orthax.hermsub(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.hermtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermtrim():
    coef = array([2, -1, 1, 0])

    pytest.raises(ValueError, beignet.orthax.hermtrim, coef, -1)

    assert_array_almost_equal(beignet.orthax.hermtrim(coef), coef[:-1])
    assert_array_almost_equal(beignet.orthax.hermtrim(coef, 1), coef[:-3])
    assert_array_almost_equal(
        beignet.orthax.hermtrim(coef, 2),
        array([0]),
    )


def test_hermval():
    assert beignet.orthax.hermval(array([]), [1]).size == 0

    x = linspace(-1, 1)
    y = [beignet.orthax.polyval(x, c) for c in hermcoefficients]

    for index in range(10):
        assert_array_almost_equal(
            beignet.orthax.hermval(
                x,
                [0] * index + [1],
            ),
            y[index],
        )

    for index in range(3):
        dims = (2,) * index
        x = zeros(dims)
        assert beignet.orthax.hermval(x, [1]).shape == dims
        assert beignet.orthax.hermval(x, array([1, 0])).shape == dims
        assert beignet.orthax.hermval(x, array([1, 0, 0])).shape == dims


def test_hermval2d():
    c1d = array([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        beignet.orthax.hermval2d,
        x1,
        x2[:2],
        c2d,
    )

    target = y1 * y2
    res = beignet.orthax.hermval2d(
        x1,
        x2,
        c2d,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = beignet.orthax.hermval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_hermval3d():
    c1d = array([2.5, 1.0, 0.75])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, beignet.orthax.hermval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    assert_array_almost_equal(
        beignet.orthax.hermval3d(x1, x2, x3, c3d),
        target,
    )

    z = ones([2, 3])
    assert beignet.orthax.hermval3d(z, z, z, c3d).shape == (2, 3)


def test_hermvander():
    x = arange(3)
    v = beignet.orthax.hermvander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], beignet.orthax.hermval(x, coef))

    x = array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.hermvander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], beignet.orthax.hermval(x, coef))


def test_hermvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(beignet.orthax.hermvander2d(x1, x2, (1, 2)), c.ravel()),
        beignet.orthax.hermval2d(x1, x2, c),
    )

    assert beignet.orthax.hermvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_hermvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(beignet.orthax.hermvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        beignet.orthax.hermval3d(x1, x2, x3, c),
    )

    assert beignet.orthax.hermvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_hermweight():
    assert_array_almost_equal(
        beignet.orthax.hermweight(linspace(-5, 5, 11)),
        exp(-(linspace(-5, 5, 11) ** 2)),
    )


def test_hermx():
    assert_array_almost_equal(beignet.orthax.hermx, array([0, 0.5]))


def test_hermzero():
    assert_array_almost_equal(beignet.orthax.hermzero, array([0]))


def test_lag2poly():
    for i in range(7):
        assert_array_almost_equal(
            beignet.orthax.lag2poly(array([0] * i + [1])), lagcoefficients[i]
        )


def test_lagadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                beignet.orthax.lagtrim(
                    beignet.orthax.lagadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.lagtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_lagcompanion():
    with pytest.raises(ValueError):
        beignet.orthax.lagcompanion(array([]))
    with pytest.raises(ValueError):
        beignet.orthax.lagcompanion([1])

    for i in range(1, 5):
        coef = array([0] * i + [1])
        assert beignet.orthax.lagcompanion(coef).shape == (i, i)

    assert beignet.orthax.lagcompanion(array([1, 2]))[0, 0] == 1.5


def test_lagder():
    pytest.raises(TypeError, beignet.orthax.lagder, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.lagder, array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.lagtrim(
                beignet.orthax.lagder(array([0] * i + [1]), order=0), tol=0.000001
            ),
            beignet.orthax.lagtrim(array([0] * i + [1]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.lagtrim(
                    beignet.orthax.lagder(
                        beignet.orthax.lagint(array([0] * i + [1]), order=j), order=j
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.lagtrim(array([0] * i + [1]), tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.lagtrim(
                    beignet.orthax.lagder(
                        beignet.orthax.lagint(array([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.lagtrim(array([0] * i + [1]), tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        beignet.orthax.lagder(c2d, axis=0),
        vstack([beignet.orthax.lagder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.lagder(
            c2d,
            axis=1,
        ),
        vstack([beignet.orthax.lagder(c) for c in c2d]),
    )


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.orthax.lagdiv(
                beignet.orthax.lagadd(
                    array([0] * i + [1]),
                    array([0] * j + [1]),
                ),
                array([0] * i + [1]),
            )

            assert_array_almost_equal(
                beignet.orthax.lagtrim(
                    beignet.orthax.lagadd(
                        beignet.orthax.lagmul(
                            quotient,
                            array([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.lagtrim(
                    beignet.orthax.lagadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagdomain():
    assert_array_almost_equal(
        beignet.orthax.lagdomain,
        array([0, 1]),
    )


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    pytest.raises(ValueError, beignet.orthax.lagfit, [1], [1], -1)
    pytest.raises(TypeError, beignet.orthax.lagfit, [[1]], [1], 0)
    pytest.raises(TypeError, beignet.orthax.lagfit, array([]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.lagfit, [1], [[[1]]], 0)
    pytest.raises(TypeError, beignet.orthax.lagfit, array([1, 2]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.lagfit, [1], array([1, 2]), 0)
    pytest.raises(TypeError, beignet.orthax.lagfit, [1], [1], 0, weight=[[1]])
    pytest.raises(TypeError, beignet.orthax.lagfit, [1], [1], 0, weight=[1, 1])
    pytest.raises(ValueError, beignet.orthax.lagfit, [1], [1], (-1,))
    pytest.raises(ValueError, beignet.orthax.lagfit, [1], [1], (2, -1, 6))
    pytest.raises(TypeError, beignet.orthax.lagfit, [1], [1], ())

    x = linspace(0, 2)
    y = f(x)

    coef3 = beignet.orthax.lagfit(
        x,
        y,
        3,
    )

    assert_array_almost_equal(
        len(coef3),
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.lagval(x, coef3),
        y,
    )

    coef3 = beignet.orthax.lagfit(
        x,
        y,
        (0, 1, 2, 3),
    )

    assert_array_almost_equal(
        len(coef3),
        4,
    )

    assert_array_almost_equal(
        beignet.orthax.lagval(x, coef3),
        y,
    )

    coef4 = beignet.orthax.lagfit(
        x,
        y,
        4,
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )

    assert_array_almost_equal(
        beignet.orthax.lagval(x, coef4),
        y,
    )

    coef4 = beignet.orthax.lagfit(
        x,
        y,
        (0, 1, 2, 3, 4),
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )

    assert_array_almost_equal(
        beignet.orthax.lagval(x, coef4),
        y,
    )

    coef2d = beignet.orthax.lagfit(
        x,
        array([y, y]).T,
        3,
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    coef2d = beignet.orthax.lagfit(
        x,
        array([y, y]).T,
        (0, 1, 2, 3),
    )

    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    w = zeros_like(x)

    yw = y.copy()

    w = w.at[1::2].set(1)
    y = y.at[0::2].set(0)

    wcoef3 = beignet.orthax.lagfit(
        x,
        yw,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef3 = beignet.orthax.lagfit(
        x,
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef2d = beignet.orthax.lagfit(
        x,
        array([yw, yw]).T,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    wcoef2d = beignet.orthax.lagfit(
        x,
        array([yw, yw]).T,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef2d,
        array([coef3, coef3]).T,
    )

    x = [1, 1j, -1, -1j]

    assert_array_almost_equal(
        beignet.orthax.lagfit(x, x, 1),
        [1, -1],
    )

    assert_array_almost_equal(
        beignet.orthax.lagfit(x, x, (0, 1)),
        [1, -1],
    )


def test_lagfromroots():
    res = beignet.orthax.lagfromroots(array([]))
    assert_array_almost_equal(
        beignet.orthax.lagtrim(res, tol=0.000001),
        array([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.orthax.lagfromroots(roots)
        res = beignet.orthax.lagval(roots, pol)
        target = 0
        assert len(pol) == i + 1
        assert_array_almost_equal(beignet.orthax.lag2poly(pol)[-1], 1)
        assert_array_almost_equal(res, target, decimal=4)


def test_laggauss():
    x, w = beignet.orthax.laggauss(100)

    v = beignet.orthax.lagvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_array_almost_equal(
        vv,
        eye(100),
    )

    target = 1.0
    assert_array_almost_equal(w.sum(), target)


def test_laggrid2d():
    c1d = array([9.0, -14.0, 6.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.laggrid2d(
            x1,
            x2,
            c2d,
        ),
        einsum("i,j->ij", y1, y2),
    )

    z = ones([2, 3])
    assert (
        beignet.orthax.laggrid2d(
            z,
            z,
            c2d,
        ).shape
        == (2, 3) * 2
    )


def test_laggrid3d():
    c1d = array([9.0, -14.0, 6.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j,k->ijk", y1, y2, y3)
    assert_array_almost_equal(
        beignet.orthax.laggrid3d(x1, x2, x3, c3d), target, decimal=3
    )

    z = ones([2, 3])
    assert beignet.orthax.laggrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_lagint():
    pytest.raises(TypeError, beignet.orthax.lagint, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.lagint, array([0]), -1)
    pytest.raises(
        ValueError,
        beignet.orthax.lagint,
        array([0]),
        1,
        array([0, 0]),
    )
    pytest.raises(ValueError, beignet.orthax.lagint, array([0]), lbnd=[0])
    pytest.raises(ValueError, beignet.orthax.lagint, array([0]), scl=[0])
    pytest.raises(TypeError, beignet.orthax.lagint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_array_almost_equal(
            beignet.orthax.lagtrim(
                beignet.orthax.lagint(array([0]), order=i, k=k), tol=0.000001
            ),
            [1, -1],
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [1 / scl]
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, order=1, k=[i])
        res = beignet.orthax.lag2poly(lagint)
        assert_array_almost_equal(
            beignet.orthax.lagtrim(res, tol=0.000001),
            beignet.orthax.lagtrim(target, tol=0.000001),
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, order=1, k=[i], lbnd=-1)
        assert_array_almost_equal(beignet.orthax.lagval(-1, lagint), i)

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [2 / scl]
        lagpol = beignet.orthax.poly2lag(pol)
        lagint = beignet.orthax.lagint(lagpol, order=1, k=[i], scl=2)
        res = beignet.orthax.lag2poly(lagint)
        assert_array_almost_equal(
            beignet.orthax.lagtrim(res, tol=0.000001),
            beignet.orthax.lagtrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = beignet.orthax.lagint(target, order=1)
            res = beignet.orthax.lagint(pol, order=j)
            assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=0.000001),
                beignet.orthax.lagtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.lagint(target, order=1, k=[k])
            res = beignet.orthax.lagint(pol, order=j, k=list(range(j)))
            assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=0.000001),
                beignet.orthax.lagtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.lagint(target, order=1, k=[k], lbnd=-1)
            res = beignet.orthax.lagint(pol, order=j, k=list(range(j)), lbnd=-1)
            assert_array_almost_equal(
                beignet.orthax.lagtrim(res, tol=0.000001),
                beignet.orthax.lagtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.lagint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                beignet.orthax.lagtrim(
                    beignet.orthax.lagint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                beignet.orthax.lagtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([beignet.orthax.lagint(c) for c in c2d.T]).T
    assert_array_almost_equal(
        beignet.orthax.lagint(c2d, axis=0),
        target,
    )

    target = vstack([beignet.orthax.lagint(c) for c in c2d])
    res = beignet.orthax.lagint(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([beignet.orthax.lagint(c, k=3) for c in c2d])
    res = beignet.orthax.lagint(
        c2d,
        k=3,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )


def test_lagline():
    assert_array_almost_equal(beignet.orthax.lagline(3, 4), [7, -4])


def test_lagmul():
    x = linspace(-3, 3, 100)

    for i in range(5):
        pol1 = array([0] * i + [1])
        val1 = beignet.orthax.lagval(x, pol1)
        for j in range(5):
            pol2 = array([0] * j + [1])
            val2 = beignet.orthax.lagval(x, pol2)
            pol3 = beignet.orthax.lagtrim(beignet.orthax.lagmul(pol1, pol2))
            val3 = beignet.orthax.lagval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_array_almost_equal(val3, val1 * val2)


def test_lagmulx():
    assert_array_almost_equal(
        beignet.orthax.lagtrim(
            beignet.orthax.lagmulx(
                [0],
            ),
            tol=0.000001,
        ),
        array([0]),
    )

    assert_array_almost_equal(
        beignet.orthax.lagtrim(
            beignet.orthax.lagmulx(
                array([1]),
            ),
            tol=0.000001,
        ),
        array([1, -1]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            beignet.orthax.lagtrim(
                beignet.orthax.lagmulx(
                    array([0] * i + [1]),
                ),
                tol=0.000001,
            ),
            beignet.orthax.lagtrim(
                array([0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]),
                tol=0.000001,
            ),
        )


def test_lagone():
    assert_array_almost_equal(
        beignet.orthax.lagone,
        array([1]),
    )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)
            assert_array_almost_equal(
                beignet.orthax.lagtrim(beignet.orthax.lagpow(c, j), tol=0.000001),
                beignet.orthax.lagtrim(
                    functools.reduce(beignet.orthax.lagmul, [c] * j, array([1])),
                    tol=0.000001,
                ),
            )


def test_lagroots():
    assert_array_almost_equal(beignet.orthax.lagroots([1]), array([]))
    assert_array_almost_equal(
        beignet.orthax.lagroots([0, 1]),
        array([1]),
    )
    for i in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.lagtrim(
                beignet.orthax.lagroots(beignet.orthax.lagfromroots(linspace(0, 3, i))),
                tol=0.000001,
            ),
            beignet.orthax.lagtrim(linspace(0, 3, i), tol=0.000001),
        )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                beignet.orthax.lagtrim(
                    beignet.orthax.lagsub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.lagtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_lagtrim():
    pytest.raises(ValueError, beignet.orthax.lagtrim, array([2, -1, 1, 0]), -1)

    assert_array_almost_equal(
        beignet.orthax.lagtrim(array([2, -1, 1, 0])), array([2, -1, 1, 0])[:-1]
    )
    assert_array_almost_equal(
        beignet.orthax.lagtrim(array([2, -1, 1, 0]), 1), array([2, -1, 1, 0])[:-3]
    )
    assert_array_almost_equal(
        beignet.orthax.lagtrim(array([2, -1, 1, 0]), 2),
        array([0]),
    )


def test_lagval():
    assert_array_almost_equal(beignet.orthax.lagval(array([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [beignet.orthax.polyval(x, c) for c in lagcoefficients]
    for i in range(7):
        assert_array_almost_equal(
            beignet.orthax.lagval(x, array([0] * i + [1])),
            y[i],
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(beignet.orthax.lagval(x, [1]).shape, dims)
        assert_array_almost_equal(beignet.orthax.lagval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(
            beignet.orthax.lagval(x, array([1, 0, 0])).shape, dims
        )


def test_lagval2d():
    c1d = array([9.0, -14.0, 6.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        beignet.orthax.lagval2d,
        x1,
        x2[:2],
        c2d,
    )

    target = y1 * y2
    assert_array_almost_equal(
        beignet.orthax.lagval2d(
            x1,
            x2,
            c2d,
        ),
        target,
        decimal=3,
    )

    z = ones([2, 3])
    assert beignet.orthax.lagval2d(
        z,
        z,
        c2d,
    ).shape == (2, 3)


def test_lagval3d():
    c1d = array([9.0, -14.0, 6.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    pytest.raises(ValueError, beignet.orthax.lagval3d, x1, x2, x3[:2], c3d)

    assert_array_almost_equal(
        beignet.orthax.lagval3d(
            x1,
            x2,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    assert beignet.orthax.lagval3d(
        ones([2, 3]), ones([2, 3]), ones([2, 3]), c3d
    ).shape == (2, 3)


def test_lagvander():
    x = arange(3)

    v = beignet.orthax.lagvander(x, 3)

    assert v.shape == (3, 4)

    for i in range(4):
        assert_array_almost_equal(
            v[..., i],
            beignet.orthax.lagval(
                x,
                array([0] * i + [1]),
            ),
        )

    x = array([[1, 2], [3, 4], [5, 6]])

    v = beignet.orthax.lagvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_array_almost_equal(
            v[..., i],
            beignet.orthax.lagval(
                x,
                array([0] * i + [1]),
            ),
        )


def test_lagvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(beignet.orthax.lagvander2d(x1, x2, (1, 2)), c.ravel()),
        beignet.orthax.lagval2d(x1, x2, c),
    )

    assert beignet.orthax.lagvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_lagvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(beignet.orthax.lagvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        beignet.orthax.lagval3d(x1, x2, x3, c),
    )

    assert beignet.orthax.lagvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_lagweight():
    assert_array_almost_equal(
        beignet.orthax.lagweight(linspace(0, 10, 11)),
        exp(-linspace(0, 10, 11)),
    )


def test_lagx():
    assert_array_almost_equal(beignet.orthax.lagx, [1, -1])


def test_lagzero():
    assert_array_almost_equal(
        beignet.orthax.lagzero,
        array([0]),
    )


def test_leg2poly():
    for index in range(10):
        assert_array_almost_equal(
            beignet.orthax.leg2poly([0] * index + [1]),
            legcoefficients[index],
        )


def test_legadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.legtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_legcompanion():
    with pytest.raises(ValueError):
        beignet.orthax.legcompanion(array([]))

    with pytest.raises(ValueError):
        beignet.orthax.legcompanion(array([1]))

    for i in range(1, 5):
        coef = array([0] * i + [1])
        assert beignet.orthax.legcompanion(coef).shape == (i, i)

    assert beignet.orthax.legcompanion(array([1, 2]))[0, 0] == -0.5


def test_legder():
    pytest.raises(TypeError, beignet.orthax.legder, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.legder, array([0]), -1)

    for i in range(5):
        target = array([0] * i + [1])
        res = beignet.orthax.legder(target, order=0)
        assert_array_almost_equal(
            beignet.orthax.legtrim(res, tol=0.000001),
            beignet.orthax.legtrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            target = array([0] * i + [1])
            res = beignet.orthax.legder(beignet.orthax.legint(target, order=j), order=j)
            assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=0.000001),
                beignet.orthax.legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            target = array([0] * i + [1])
            res = beignet.orthax.legder(
                beignet.orthax.legint(target, order=j, scl=2), order=j, scl=0.5
            )
            assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=0.000001),
                beignet.orthax.legtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([beignet.orthax.legder(c) for c in c2d.T]).T
    res = beignet.orthax.legder(c2d, axis=0)
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([beignet.orthax.legder(c) for c in c2d])
    res = beignet.orthax.legder(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    c = (1, 2, 3, 4)
    assert_array_almost_equal(
        beignet.orthax.legder(c, 4),
        array([0]),
    )


def test_legdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.orthax.legdiv(
                beignet.orthax.legadd(
                    array([0] * i + [1]),
                    array([0] * j + [1]),
                ),
                array([0] * i + [1]),
            )

            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legadd(
                        beignet.orthax.legmul(
                            quotient,
                            array([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.legtrim(
                    beignet.orthax.legadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legdomain():
    assert_array_almost_equal(beignet.orthax.legdomain, [-1, 1])


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    pytest.raises(ValueError, beignet.orthax.legfit, [1], [1], -1)
    pytest.raises(TypeError, beignet.orthax.legfit, [[1]], [1], 0)
    pytest.raises(TypeError, beignet.orthax.legfit, array([]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.legfit, [1], [[[1]]], 0)
    pytest.raises(TypeError, beignet.orthax.legfit, array([1, 2]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.legfit, [1], array([1, 2]), 0)
    pytest.raises(TypeError, beignet.orthax.legfit, [1], [1], 0, weight=[[1]])
    pytest.raises(TypeError, beignet.orthax.legfit, [1], [1], 0, weight=[1, 1])
    pytest.raises(ValueError, beignet.orthax.legfit, [1], [1], (-1,))
    pytest.raises(ValueError, beignet.orthax.legfit, [1], [1], (2, -1, 6))
    pytest.raises(TypeError, beignet.orthax.legfit, [1], [1], ())

    coef3 = beignet.orthax.legfit(linspace(0, 2), f(linspace(0, 2)), 3)
    assert_array_almost_equal(len(coef3), 4)
    assert_array_almost_equal(
        beignet.orthax.legval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )
    coef3 = beignet.orthax.legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3),
    )
    assert_array_almost_equal(
        len(coef3),
        4,
    )
    assert_array_almost_equal(
        beignet.orthax.legval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )

    coef4 = beignet.orthax.legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        4,
    )
    assert_array_almost_equal(
        len(coef4),
        5,
    )
    assert_array_almost_equal(
        beignet.orthax.legval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )
    coef4 = beignet.orthax.legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3, 4),
    )
    assert_array_almost_equal(
        len(coef4),
        5,
    )
    assert_array_almost_equal(
        beignet.orthax.legval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )

    coef4 = beignet.orthax.legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (2, 3, 4, 1, 0),
    )

    assert_array_almost_equal(
        len(coef4),
        5,
    )

    assert_array_almost_equal(
        beignet.orthax.legval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(0, 2),
            array([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
            3,
        ),
        array([coef3, coef3]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(0, 2),
            array([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
            (0, 1, 2, 3),
        ),
        array([coef3, coef3]).T,
    )

    w = zeros_like(linspace(0, 2))

    yw = f(linspace(0, 2)).copy()

    w = w.at[1::2].set(1)

    i = f(linspace(0, 2))

    i = i.at[0::2].set(0)

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(0, 2),
            yw,
            3,
            weight=w,
        ),
        coef3,
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(0, 2),
            yw,
            (0, 1, 2, 3),
            weight=w,
        ),
        coef3,
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(0, 2),
            array([yw, yw]).T,
            3,
            weight=w,
        ),
        array([coef3, coef3]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(0, 2),
            array([yw, yw]).T,
            (0, 1, 2, 3),
            weight=w,
        ),
        array([coef3, coef3]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            [1, 1j, -1, -1j],
            [1, 1j, -1, -1j],
            1,
        ),
        [0, 1],
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            [1, 1j, -1, -1j],
            [1, 1j, -1, -1j],
            (0, 1),
        ),
        [0, 1],
    )

    assert_array_almost_equal(
        beignet.orthax.legval(
            linspace(-1, 1),
            beignet.orthax.legfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                4,
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_array_almost_equal(
        beignet.orthax.legval(
            linspace(-1, 1),
            beignet.orthax.legfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                (0, 2, 4),
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_array_almost_equal(
        beignet.orthax.legfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            4,
        ),
        beignet.orthax.legfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            (0, 2, 4),
        ),
    )


def test_legfromroots():
    assert_array_almost_equal(
        beignet.orthax.legtrim(beignet.orthax.legfromroots(array([])), tol=0.000001),
        [1],
    )
    for i in range(1, 5):
        assert (
            beignet.orthax.legfromroots(
                cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
            ).shape[-1]
            == i + 1
        )
        assert_array_almost_equal(
            beignet.orthax.leg2poly(
                beignet.orthax.legfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2]))
            )[-1],
            1,
        )
        assert_array_almost_equal(
            beignet.orthax.legval(
                cos(linspace(-math.pi, 0, 2 * i + 1)[1::2]),
                beignet.orthax.legfromroots(
                    cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
                ),
            ),
            0,
        )


def test_leggauss():
    x, w = beignet.orthax.leggauss(100)

    v = beignet.orthax.legvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd

    assert_array_almost_equal(
        vv,
        eye(100),
    )

    assert_array_almost_equal(w.sum(), 2.0)


def test_leggrid2d():
    c1d = array([2.0, 2.0, 2.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.leggrid2d(
            x1,
            x2,
            c2d,
        ),
        einsum("i,j->ij", y1, y2),
    )

    z = ones([2, 3])
    assert (
        beignet.orthax.leggrid2d(
            z,
            z,
            c2d,
        ).shape
        == (2, 3) * 2
    )


def test_leggrid3d():
    c1d = array([2.0, 2.0, 2.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.leggrid3d(
            x1,
            x2,
            x3,
            c3d,
        ),
        einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    assert (
        beignet.orthax.leggrid3d(ones([2, 3]), ones([2, 3]), ones([2, 3]), c3d).shape
        == (2, 3) * 3
    )


def test_legint():
    pytest.raises(TypeError, beignet.orthax.legint, array([0]), 0.5)
    pytest.raises(ValueError, beignet.orthax.legint, array([0]), -1)
    pytest.raises(
        ValueError,
        beignet.orthax.legint,
        array([0]),
        1,
        array([0, 0]),
    )
    pytest.raises(ValueError, beignet.orthax.legint, array([0]), lbnd=[0])
    pytest.raises(ValueError, beignet.orthax.legint, array([0]), scl=[0])
    pytest.raises(TypeError, beignet.orthax.legint, array([0]), axis=0.5)

    for i in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.legtrim(
                beignet.orthax.legint(array([0]), order=i, k=([0] * (i - 2) + [1])),
                tol=0.000001,
            ),
            [0, 1],
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.legtrim(
                beignet.orthax.leg2poly(
                    beignet.orthax.legint(
                        beignet.orthax.poly2leg(array([0] * i + [1])),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            beignet.orthax.legtrim(array([i] + [0] * i + [1 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.legval(
                -1,
                beignet.orthax.legint(
                    beignet.orthax.poly2leg(array([0] * i + [1])),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.legtrim(
                beignet.orthax.leg2poly(
                    beignet.orthax.legint(
                        beignet.orthax.poly2leg(array([0] * i + [1])),
                        order=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tol=0.000001,
            ),
            beignet.orthax.legtrim(array([i] + [0] * i + [2 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            target = (array([0] * i + [1]))[:]
            for _ in range(j):
                target = beignet.orthax.legint(target, order=1)
            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legint(array([0] * i + [1]), order=j), tol=0.000001
                ),
                beignet.orthax.legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.legint(target, order=1, k=[k])
            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legint(pol, order=j, k=list(range(j))), tol=0.000001
                ),
                beignet.orthax.legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.legint(target, order=1, k=[k], lbnd=-1)
            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legint(pol, order=j, k=list(range(j)), lbnd=-1),
                    tol=0.000001,
                ),
                beignet.orthax.legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = beignet.orthax.legint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                beignet.orthax.legtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        beignet.orthax.legint(c2d, axis=0),
        vstack([beignet.orthax.legint(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.legint(
            c2d,
            axis=1,
        ),
        vstack([beignet.orthax.legint(c) for c in c2d]),
    )

    assert_array_almost_equal(
        beignet.orthax.legint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([beignet.orthax.legint(c, k=3) for c in c2d]),
    )

    assert_array_almost_equal(beignet.orthax.legint((1, 2, 3), 0), (1, 2, 3))


def test_legline():
    assert_array_almost_equal(beignet.orthax.legline(3, 4), array([3, 4]))

    assert_array_almost_equal(
        beignet.orthax.legtrim(
            beignet.orthax.legline(3, 0),
            tol=0.000001,
        ),
        [3],
    )


def test_legmul():
    for i in range(5):
        pol1 = array([0] * i + [1])
        x = linspace(-1, 1, 100)
        val1 = beignet.orthax.legval(x, pol1)
        for j in range(5):
            pol2 = array([0] * j + [1])
            val2 = beignet.orthax.legval(x, pol2)
            pol3 = beignet.orthax.legmul(pol1, pol2)
            val3 = beignet.orthax.legval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_array_almost_equal(val3, val1 * val2)


def test_legmulx():
    assert_array_almost_equal(
        beignet.orthax.legtrim(
            beignet.orthax.legmulx(
                array([0]),
            ),
            tol=0.000001,
        ),
        array([0]),
    )

    assert_array_almost_equal(
        beignet.orthax.legtrim(
            beignet.orthax.legmulx(
                [1],
            ),
            tol=0.000001,
        ),
        [0, 1],
    )

    for index in range(1, 5):
        tmp = 2 * index + 1

        assert_array_almost_equal(
            beignet.orthax.legtrim(
                beignet.orthax.legmulx(
                    [0] * index + [1],
                ),
                tol=0.000001,
            ),
            [0] * (index - 1) + [index / tmp, 0, (index + 1) / tmp],
        )


def test_legone():
    assert_array_almost_equal(
        beignet.orthax.legone,
        array([1]),
    )


def test_legpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)

            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legpow(
                        c,
                        j,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.legtrim(
                    functools.reduce(
                        beignet.orthax.legmul,
                        [c] * j,
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legroots():
    assert_array_almost_equal(beignet.orthax.legroots([1]), array([]))
    assert_array_almost_equal(beignet.orthax.legroots(array([1, 2])), array([-0.5]))

    for index in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.legtrim(
                beignet.orthax.legroots(
                    beignet.orthax.legfromroots(
                        linspace(-1, 1, index),
                    ),
                ),
                tol=0.000001,
            ),
            beignet.orthax.legtrim(
                linspace(-1, 1, index),
                tol=0.000001,
            ),
        )


def test_legsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                beignet.orthax.legtrim(
                    beignet.orthax.legsub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.legtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_legtrim():
    pytest.raises(ValueError, beignet.orthax.legtrim, array([2, -1, 1, 0]), -1)

    assert_array_almost_equal(
        beignet.orthax.legtrim(array([2, -1, 1, 0])), array([2, -1, 1, 0])[:-1]
    )
    assert_array_almost_equal(
        beignet.orthax.legtrim(array([2, -1, 1, 0]), 1), array([2, -1, 1, 0])[:-3]
    )
    assert_array_almost_equal(
        beignet.orthax.legtrim(array([2, -1, 1, 0]), 2),
        array([0]),
    )


def test_legval():
    assert_array_almost_equal(beignet.orthax.legval(array([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [beignet.orthax.polyval(x, c) for c in legcoefficients]
    for i in range(10):
        assert_array_almost_equal(beignet.orthax.legval(x, array([0] * i + [1])), y[i])

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(beignet.orthax.legval(x, [1]).shape, dims)
        assert_array_almost_equal(beignet.orthax.legval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(
            beignet.orthax.legval(x, array([1, 0, 0])).shape, dims
        )


def test_legval2d():
    c1d = array([2.0, 2.0, 2.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        beignet.orthax.legval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_array_almost_equal(
        beignet.orthax.legval2d(
            x1,
            x2,
            c2d,
        ),
        y1 * y2,
    )

    z = ones([2, 3])
    assert beignet.orthax.legval2d(
        z,
        z,
        c2d,
    ).shape == (2, 3)


def test_legval3d():
    c1d = array([2.0, 2.0, 2.0])
    c3d = einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, array([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, beignet.orthax.legval3d, x1, x2, x3[:2], c3d)

    assert_array_almost_equal(
        beignet.orthax.legval3d(
            x1,
            x2,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    z = ones([2, 3])
    assert beignet.orthax.legval3d(z, z, z, c3d).shape == (2, 3)


def test_legvander():
    x = arange(3)
    v = beignet.orthax.legvander(x, 3)
    assert v.shape == (3, 4)

    for index in range(4):
        assert_array_almost_equal(
            v[..., index],
            beignet.orthax.legval(
                x,
                [0] * index + [1],
            ),
        )

    x = array([[1, 2], [3, 4], [5, 6]])
    v = beignet.orthax.legvander(x, 3)
    assert v.shape == (3, 2, 4)

    for index in range(4):
        assert_array_almost_equal(
            v[..., index],
            beignet.orthax.legval(
                x,
                [0] * index + [1],
            ),
        )

    pytest.raises(ValueError, beignet.orthax.legvander, (1, 2, 3), -1)


def test_legvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(beignet.orthax.legvander2d(x1, x2, (1, 2)), c.ravel()),
        beignet.orthax.legval2d(x1, x2, c),
    )

    assert beignet.orthax.legvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_legvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(beignet.orthax.legvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        beignet.orthax.legval3d(x1, x2, x3, c),
    )

    assert beignet.orthax.legvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_legweight():
    assert_array_almost_equal(beignet.orthax.legweight(linspace(-1, 1, 11)), 1.0)


def test_legx():
    assert_array_almost_equal(
        beignet.orthax.legx,
        array([0, 1]),
    )


def test_legzero():
    assert_array_almost_equal(
        beignet.orthax.legzero,
        array([0]),
    )


def test_poly2cheb():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.poly2cheb(
                chebcoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_poly2herm():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.hermtrim(
                beignet.orthax.poly2herm(
                    hermcoefficients[i],
                ),
                tol=0.000001,
            ),
            array([0] * i + [1]),
        )


def test_poly2herme():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.poly2herme(
                hermecoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_poly2lag():
    for i in range(7):
        assert_array_almost_equal(
            beignet.orthax.poly2lag(
                lagcoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_poly2leg():
    for i in range(10):
        assert_array_almost_equal(
            beignet.orthax.poly2leg(
                legcoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_polyadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polycompanion():
    with pytest.raises(ValueError):
        beignet.orthax.polycompanion(array([]))

    with pytest.raises(ValueError):
        beignet.orthax.polycompanion(array([1]))

    for i in range(1, 5):
        output = beignet.orthax.polycompanion(
            array([0] * i + [1]),
        )

        assert output.shape == (i, i)

    output = beignet.orthax.polycompanion(
        array([1, 2]),
    )

    assert output[0, 0] == -0.5


def test_polyder():
    with pytest.raises(TypeError):
        beignet.orthax.polyder(array([0]), axis=0.5)

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.polytrim(
                beignet.orthax.polyder(
                    array([0] * i + [1]),
                    order=0,
                ),
                tol=0.000001,
            ),
            beignet.orthax.polytrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyder(
                        beignet.orthax.polyint(
                            array([0] * i + [1]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyder(
                        beignet.orthax.polyint(
                            array([0] * i + [1]),
                            order=j,
                            scl=2,
                        ),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        beignet.orthax.polyder(
            c2d,
            axis=0,
        ),
        vstack([beignet.orthax.polyder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.polyder(
            c2d,
            axis=1,
        ),
        vstack([beignet.orthax.polyder(c) for c in c2d]),
    )


def test_polydiv():
    quotient, remainder = beignet.orthax.polydiv(
        array([2]),
        array([2]),
    )

    assert_array_almost_equal(
        quotient,
        array([1]),
    )

    assert_array_almost_equal(
        remainder,
        array([0]),
    )

    quotient, remainder = beignet.orthax.polydiv(
        array([2, 2]),
        array([2]),
    )

    assert_array_almost_equal(
        quotient,
        array([1, 1]),
    )

    assert_array_almost_equal(
        remainder,
        array([0]),
    )

    for i in range(5):
        for j in range(5):
            target = beignet.orthax.polyadd(
                array([0.0] * i + [1.0, 2.0]),
                array([0.0] * j + [1.0, 2.0]),
            )

            quotient, remainder = beignet.orthax.polydiv(
                target,
                array([0.0] * i + [1.0, 2.0]),
            )

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyadd(
                        beignet.orthax.polymul(
                            quotient,
                            array([0.0] * i + [1.0, 2.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polydomain():
    assert_array_almost_equal(
        beignet.orthax.polydomain,
        array([-1, 1]),
    )


def test_polyfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    pytest.raises(ValueError, beignet.orthax.polyfit, [1], [1], -1)
    pytest.raises(TypeError, beignet.orthax.polyfit, [[1]], [1], 0)
    pytest.raises(TypeError, beignet.orthax.polyfit, array([]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.polyfit, [1], [[[1]]], 0)
    pytest.raises(TypeError, beignet.orthax.polyfit, array([1, 2]), [1], 0)
    pytest.raises(TypeError, beignet.orthax.polyfit, [1], array([1, 2]), 0)
    pytest.raises(TypeError, beignet.orthax.polyfit, [1], [1], 0, weight=[[1]])
    pytest.raises(TypeError, beignet.orthax.polyfit, [1], [1], 0, weight=[1, 1])
    pytest.raises(ValueError, beignet.orthax.polyfit, [1], [1], (-1,))
    pytest.raises(ValueError, beignet.orthax.polyfit, [1], [1], (2, -1, 6))
    pytest.raises(TypeError, beignet.orthax.polyfit, [1], [1], ())

    coef3 = beignet.orthax.polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        3,
    )
    assert coef3.shape[0] == 4
    assert_array_almost_equal(
        beignet.orthax.polyval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )
    coef3 = beignet.orthax.polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3),
    )
    assert len(coef3) == 4
    assert_array_almost_equal(
        beignet.orthax.polyval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )

    coef4 = beignet.orthax.polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        4,
    )
    assert len(coef4) == 5
    assert_array_almost_equal(
        beignet.orthax.polyval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )
    coef4 = beignet.orthax.polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3, 4),
    )
    assert len(coef4) == 5
    assert_array_almost_equal(
        beignet.orthax.polyval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )

    coef2d = beignet.orthax.polyfit(
        linspace(0, 2),
        array([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
        3,
    )
    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )
    coef2d = beignet.orthax.polyfit(
        linspace(0, 2),
        array([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
        (0, 1, 2, 3),
    )
    assert_array_almost_equal(
        coef2d,
        array([coef3, coef3]).T,
    )

    w = zeros_like(linspace(0, 2))

    w = w.at[1::2].set(1)
    yw = f(linspace(0, 2)).at[0::2].set(0)

    wcoef3 = beignet.orthax.polyfit(
        linspace(0, 2),
        yw,
        3,
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    wcoef3 = beignet.orthax.polyfit(
        linspace(0, 2),
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_array_almost_equal(
        wcoef3,
        coef3,
    )

    assert_array_almost_equal(
        beignet.orthax.polyfit(
            linspace(0, 2),
            array([yw, yw]).T,
            3,
            weight=w,
        ),
        array([coef3, coef3]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.polyfit(
            linspace(0, 2),
            array([yw, yw]).T,
            (0, 1, 2, 3),
            weight=w,
        ),
        array([coef3, coef3]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.polyfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            1,
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            (0, 1),
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyval(
            linspace(-1, 1),
            beignet.orthax.polyfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                4,
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_array_almost_equal(
        beignet.orthax.polyval(
            linspace(-1, 1),
            beignet.orthax.polyfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                (0, 2, 4),
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_array_almost_equal(
        beignet.orthax.polyfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            4,
        ),
        beignet.orthax.polyfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            (0, 2, 4),
        ),
    )


def test_polyfromroots():
    assert_array_almost_equal(
        beignet.orthax.polytrim(
            beignet.orthax.polyfromroots(array([])),
            tol=0.000001,
        ),
        array([1]),
    )

    for i in range(1, 5):
        roots = cos(
            linspace(-math.pi, 0, 2 * i + 1)[1::2],
        )

        assert_array_almost_equal(
            beignet.orthax.polytrim(
                beignet.orthax.polyfromroots(roots) * 2 ** (i - 1),
                tol=0.000001,
            ),
            beignet.orthax.polytrim(
                polycoefficients[i],
                tol=0.000001,
            ),
        )


def test_polygrid2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    x1, x2, x3 = x
    y1, y2, y3 = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    assert_array_almost_equal(
        beignet.orthax.polygrid2d(
            x1,
            x2,
            einsum(
                "i,j->ij",
                array([1.0, 2.0, 3.0]),
                array([1.0, 2.0, 3.0]),
            ),
        ),
        einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    output = beignet.orthax.polygrid2d(
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j->ij",
            array([1.0, 2.0, 3.0]),
            array([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 2


def test_polygrid3d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = beignet.orthax.polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    assert_array_almost_equal(
        beignet.orthax.polygrid3d(
            x1,
            x2,
            x3,
            einsum(
                "i,j,k->ijk",
                array([1.0, 2.0, 3.0]),
                array([1.0, 2.0, 3.0]),
                array([1.0, 2.0, 3.0]),
            ),
        ),
        einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    output = beignet.orthax.polygrid3d(
        ones([2, 3]),
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j,k->ijk",
            array([1.0, 2.0, 3.0]),
            array([1.0, 2.0, 3.0]),
            array([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_polyint():
    with pytest.raises(TypeError):
        beignet.orthax.polyint(array([0]), 0.5)

    with pytest.raises(ValueError):
        beignet.orthax.polyint(array([0]), -1)

    with pytest.raises(ValueError):
        beignet.orthax.polyint(
            array([0]),
            1,
            array([0, 0]),
        )

    with pytest.raises(ValueError):
        beignet.orthax.polyint(array([0]), lbnd=[0])

    with pytest.raises(ValueError):
        beignet.orthax.polyint(array([0]), scl=[0])

    with pytest.raises(TypeError):
        beignet.orthax.polyint(array([0]), axis=0.5)

    for i in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.polytrim(
                beignet.orthax.polyint(
                    array([0]),
                    order=i,
                    k=([0] * (i - 2) + [1]),
                ),
                tol=0.000001,
            ),
            [0, 1],
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.polytrim(
                beignet.orthax.polyint(
                    array([0] * i + [1]),
                    order=1,
                    k=[i],
                ),
                tol=0.000001,
            ),
            beignet.orthax.polytrim(
                array([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.polyval(
                -1,
                beignet.orthax.polyint(
                    array([0] * i + [1]),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.polytrim(
                beignet.orthax.polyint(
                    array([0] * i + [1]),
                    order=1,
                    k=[i],
                    scl=2,
                ),
                tol=0.000001,
            ),
            beignet.orthax.polytrim(
                array([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for _ in range(j):
                target = beignet.orthax.polyint(
                    target,
                    order=1,
                )

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyint(
                        pol,
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = beignet.orthax.polyint(
                    target,
                    order=1,
                    k=[k],
                )

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = beignet.orthax.polyint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = beignet.orthax.polyint(
                    target,
                    order=1,
                    k=[k],
                    scl=2,
                )

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scl=2,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 6))

    assert_array_almost_equal(
        beignet.orthax.polyint(
            c2d,
            axis=0,
        ),
        vstack([beignet.orthax.polyint(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        beignet.orthax.polyint(
            c2d,
            axis=1,
        ),
        vstack([beignet.orthax.polyint(c) for c in c2d]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([beignet.orthax.polyint(c, k=3) for c in c2d]),
    )


def test_polyline():
    assert_array_almost_equal(
        beignet.orthax.polyline(3, 4),
        array([3, 4]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyline(3, 0),
        array([3, 0]),
    )


def test_polymul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[i + j].set(target[i + j] + 1)

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polymul(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polymulx():
    assert_array_almost_equal(
        beignet.orthax.polymulx(
            array([0]),
        ),
        array([0, 0]),
    )

    assert_array_almost_equal(
        beignet.orthax.polymulx(
            array([1]),
        ),
        array([0, 1]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            beignet.orthax.polymulx(
                array([0] * i + [1]),
            ),
            array([0] * (i + 1) + [1]),
        )


def test_polyone():
    assert_array_almost_equal(
        beignet.orthax.polyone,
        array([1]),
    )


def test_polypow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polypow(
                        arange(i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    functools.reduce(
                        beignet.orthax.polymul,
                        [(arange(i + 1))] * j,
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_polyroots():
    assert_array_almost_equal(
        beignet.orthax.polyroots([1]),
        array([]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyroots(array([1, 2])),
        array([-0.5]),
    )

    for i in range(2, 5):
        assert_array_almost_equal(
            beignet.orthax.polytrim(
                beignet.orthax.polyroots(
                    beignet.orthax.polyfromroots(
                        linspace(-1, 1, i),
                    ),
                ),
                tol=0.000001,
            ),
            beignet.orthax.polytrim(
                linspace(-1, 1, i),
                tol=0.000001,
            ),
        )


def test_polysub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                beignet.orthax.polytrim(
                    beignet.orthax.polysub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                beignet.orthax.polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polytrim():
    coef = array([2, -1, 1, 0])

    pytest.raises(ValueError, beignet.orthax.polytrim, coef, -1)

    assert_array_almost_equal(
        beignet.orthax.polytrim(coef),
        coef[:-1],
    )

    assert_array_almost_equal(
        beignet.orthax.polytrim(coef, 1),
        coef[:-3],
    )

    assert_array_almost_equal(
        beignet.orthax.polytrim(coef, 2),
        array([0]),
    )


def test_polyval():
    assert beignet.orthax.polyval(array([]), [1]).size == 0

    x = linspace(-1, 1)
    y = [x**i for i in range(5)]

    for i in range(5):
        assert_array_almost_equal(
            beignet.orthax.polyval(x, array([0] * i + [1])),
            y[i],
        )

    assert_array_almost_equal(
        beignet.orthax.polyval(x, [0, -1, 0, 1]),
        x * (x**2 - 1),
    )

    for i in range(3):
        dims = (2,) * i
        x = zeros(dims)

        assert beignet.orthax.polyval(x, array([1])).shape == dims

        assert beignet.orthax.polyval(x, array([1, 0])).shape == dims

        assert beignet.orthax.polyval(x, array([1, 0, 0])).shape == dims


def test_polyval2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    x1, x2, x3 = x

    y1, y2, y3 = beignet.orthax.polyval(
        x,
        array([1.0, 2.0, 3.0]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyval2d(
            x1,
            x2,
            einsum(
                "i,j->ij",
                array([1.0, 2.0, 3.0]),
                array([1.0, 2.0, 3.0]),
            ),
        ),
        y1 * y2,
    )

    output = beignet.orthax.polyval2d(
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j->ij",
            array([1.0, 2.0, 3.0]),
            array([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_polyval3d():
    x = jax.random.uniform(
        key,
        (3, 5),
        minval=-1,
        maxval=1,
    )

    x1, x2, x3 = x

    y1, y2, y3 = beignet.orthax.polyval(
        x,
        array([1.0, 2.0, 3.0]),
    )

    assert_array_almost_equal(
        beignet.orthax.polyval3d(
            x1,
            x2,
            x3,
            einsum(
                "i,j,k->ijk",
                array([1.0, 2.0, 3.0]),
                array([1.0, 2.0, 3.0]),
                array([1.0, 2.0, 3.0]),
            ),
        ),
        y1 * y2 * y3,
    )

    output = beignet.orthax.polyval3d(
        ones([2, 3]),
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j,k->ijk",
            array([1.0, 2.0, 3.0]),
            array([1.0, 2.0, 3.0]),
            array([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_polyvalfromroots():
    pytest.raises(
        ValueError,
        beignet.orthax.polyvalfromroots,
        array([1]),
        array([1]),
        tensor=False,
    )

    assert beignet.orthax.polyvalfromroots(array([]), array([1])).size == 0

    assert beignet.orthax.polyvalfromroots(array([]), array([1])).shape == (0,)

    assert beignet.orthax.polyvalfromroots(array([]), [[1] * 5]).size == 0

    assert beignet.orthax.polyvalfromroots(array([]), [[1] * 5]).shape == (5, 0)

    assert_array_almost_equal(
        beignet.orthax.polyvalfromroots(1, 1),
        0,
    )

    assert beignet.orthax.polyvalfromroots(1, ones((3, 3))).shape == (3,)

    y = [linspace(-1, 1) ** i for i in range(5)]
    for i in range(1, 5):
        target = y[i]
        res = beignet.orthax.polyvalfromroots(linspace(-1, 1), [0] * i)
        assert_array_almost_equal(
            res,
            target,
        )
    target = linspace(-1, 1) * (linspace(-1, 1) - 1) * (linspace(-1, 1) + 1)
    res = beignet.orthax.polyvalfromroots(linspace(-1, 1), array([-1, 0, 1]))
    assert_array_almost_equal(
        res,
        target,
    )

    for i in range(3):
        dims = (2,) * i
        x = zeros(dims)
        assert beignet.orthax.polyvalfromroots(x, array([1])).shape == dims
        assert beignet.orthax.polyvalfromroots(x, array([1, 0])).shape == dims
        assert beignet.orthax.polyvalfromroots(x, array([1, 0, 0])).shape == dims

    ptest = array([15, 2, -16, -2, 1])
    r = beignet.orthax.polyroots(ptest)
    assert_array_almost_equal(
        beignet.orthax.polyval(linspace(-1, 1), ptest),
        beignet.orthax.polyvalfromroots(linspace(-1, 1), r),
    )

    rshape = (3, 5)

    x = arange(-3, 2)

    r = jax.random.randint(key, rshape, -5, 5)

    target = empty(r.shape[1:])

    for ii in range(target.size):
        target = target.at[ii].set(beignet.orthax.polyvalfromroots(x[ii], r[:, ii]))

    assert_array_almost_equal(
        beignet.orthax.polyvalfromroots(x, r, tensor=False),
        target,
    )

    x = vstack([x, 2 * x])

    target = empty(r.shape[1:] + x.shape)

    for ii in range(r.shape[1]):
        for jj in range(x.shape[0]):
            target = target.at[ii, jj, :].set(
                beignet.orthax.polyvalfromroots(
                    x[jj],
                    r[:, ii],
                )
            )

    assert_array_almost_equal(
        beignet.orthax.polyvalfromroots(x, r, tensor=True),
        target,
    )


def test_polyvander():
    x = arange(3)

    v = beignet.orthax.polyvander(x, 3)

    assert v.shape == (3, 4)

    for i in range(4):
        assert_array_almost_equal(
            v[..., i],
            beignet.orthax.polyval(
                x,
                array([0] * i + [1]),
            ),
        )

    x = array([[1, 2], [3, 4], [5, 6]])

    v = beignet.orthax.polyvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_array_almost_equal(
            v[..., i],
            beignet.orthax.polyval(
                x,
                array([0] * i + [1]),
            ),
        )

    pytest.raises(
        ValueError,
        beignet.orthax.polyvander,
        arange(3),
        -1,
    )


def test_polyvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    c = jax.random.uniform(key, (2, 3))

    assert_array_almost_equal(
        dot(
            beignet.orthax.polyvander2d(
                x1,
                x2,
                (1, 2),
            ),
            c.ravel(),
        ),
        beignet.orthax.polyval2d(
            x1,
            x2,
            c,
        ),
    )

    van = beignet.orthax.polyvander2d(
        [x1],
        [x2],
        (1, 2),
    )

    assert van.shape == (1, 5, 6)


def test_polyvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    c = jax.random.uniform(key, (2, 3, 4))

    assert_array_almost_equal(
        dot(
            beignet.orthax.polyvander3d(
                x1,
                x2,
                x3,
                (1, 2, 3),
            ),
            c.ravel(),
        ),
        beignet.orthax.polyval3d(
            x1,
            x2,
            x3,
            c,
        ),
    )

    van = beignet.orthax.polyvander3d(
        [x1],
        [x2],
        [x3],
        (1, 2, 3),
    )

    assert van.shape == (1, 5, 24)


def test_polyx():
    assert_array_almost_equal(
        beignet.orthax.polyx,
        array([0, 1]),
    )


def test_polyzero():
    assert_array_almost_equal(
        beignet.orthax.polyzero,
        array([0]),
    )
