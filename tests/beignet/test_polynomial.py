import functools
import math

import jax
import numpy
import pytest
from beignet.polynomial import (
    _c_series_to_z_series,
    _fit,
    _get_domain,
    _map_domain,
    _map_parameters,
    _pow,
    _trim_coefficients,
    _trim_sequence,
    _vandermonde,
    _z_series_to_c_series,
    cheb2poly,
    chebadd,
    chebcompanion,
    chebder,
    chebdiv,
    chebdomain,
    chebfit,
    chebfromroots,
    chebgauss,
    chebgrid2d,
    chebgrid3d,
    chebint,
    chebinterpolate,
    chebline,
    chebmul,
    chebmulx,
    chebone,
    chebpow,
    chebpts1,
    chebpts2,
    chebroots,
    chebsub,
    chebtrim,
    chebval,
    chebval2d,
    chebval3d,
    chebvander,
    chebvander2d,
    chebvander3d,
    chebweight,
    chebx,
    chebzero,
    herm2poly,
    hermadd,
    hermcompanion,
    hermder,
    hermdiv,
    hermdomain,
    herme2poly,
    hermeadd,
    hermecompanion,
    hermeder,
    hermediv,
    hermedomain,
    hermefit,
    hermefromroots,
    hermegauss,
    hermegrid2d,
    hermegrid3d,
    hermeint,
    hermeline,
    hermemul,
    hermemulx,
    hermeone,
    hermepow,
    hermeroots,
    hermesub,
    hermetrim,
    hermeval,
    hermeval2d,
    hermeval3d,
    hermevander,
    hermevander2d,
    hermevander3d,
    hermeweight,
    hermex,
    hermezero,
    hermfit,
    hermfromroots,
    hermgauss,
    hermgrid2d,
    hermgrid3d,
    hermint,
    hermline,
    hermmul,
    hermmulx,
    hermone,
    hermpow,
    hermroots,
    hermsub,
    hermtrim,
    hermval,
    hermval2d,
    hermval3d,
    hermvander,
    hermvander2d,
    hermvander3d,
    hermweight,
    hermx,
    hermzero,
    lag2poly,
    lagadd,
    lagcompanion,
    lagder,
    lagdiv,
    lagdomain,
    lagfit,
    lagfromroots,
    laggauss,
    laggrid2d,
    laggrid3d,
    lagint,
    lagline,
    lagmul,
    lagmulx,
    lagone,
    lagpow,
    lagroots,
    lagsub,
    lagtrim,
    lagval,
    lagval2d,
    lagval3d,
    lagvander,
    lagvander2d,
    lagvander3d,
    lagweight,
    lagx,
    lagzero,
    leg2poly,
    legadd,
    legcompanion,
    legder,
    legdiv,
    legdomain,
    legfit,
    legfromroots,
    leggauss,
    leggrid2d,
    leggrid3d,
    legint,
    legline,
    legmul,
    legmulx,
    legone,
    legpow,
    legroots,
    legsub,
    legtrim,
    legval,
    legval2d,
    legval3d,
    legvander,
    legvander2d,
    legvander3d,
    legweight,
    legx,
    legzero,
    poly2cheb,
    poly2herm,
    poly2herme,
    poly2lag,
    poly2leg,
    polyadd,
    polycompanion,
    polyder,
    polydiv,
    polydomain,
    polyfit,
    polyfromroots,
    polygrid2d,
    polygrid3d,
    polyint,
    polyline,
    polymul,
    polymulx,
    polyone,
    polypow,
    polyroots,
    polysub,
    polytrim,
    polyval,
    polyval2d,
    polyval3d,
    polyvalfromroots,
    polyvander,
    polyvander2d,
    polyvander3d,
    polyx,
    polyzero,
    sum,
    transpose,
)
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
    ravel,
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
    array([1]),
    array([0, 1]),
    array([-1, 0, 2]),
    array([0, -3, 0, 4]),
    array([1, 0, -8, 0, 8]),
    array([0, 5, 0, -20, 0, 16]),
    array([-1, 0, 18, 0, -48, 0, 32]),
    array([0, -7, 0, 56, 0, -112, 0, 64]),
    array([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    array([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]

hermcoefficients = [
    array([1]),
    array([0, 2]),
    array([-2, 0, 4]),
    array([0, -12, 0, 8]),
    array([12, 0, -48, 0, 16]),
    array([0, 120, 0, -160, 0, 32]),
    array([-120, 0, 720, 0, -480, 0, 64]),
    array([0, -1680, 0, 3360, 0, -1344, 0, 128]),
    array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
    array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
]

hermecoefficients = [
    array([1]),
    array([0, 1]),
    array([-1, 0, 1]),
    array([0, -3, 0, 1]),
    array([3, 0, -6, 0, 1]),
    array([0, 15, 0, -10, 0, 1]),
    array([-15, 0, 45, 0, -15, 0, 1]),
    array([0, -105, 0, 105, 0, -21, 0, 1]),
    array([105, 0, -420, 0, 210, 0, -28, 0, 1]),
    array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
]

lagcoefficients = [
    array([1]) / 1,
    array([1, -1]) / 1,
    array([2, -4, 1]) / 2,
    array([6, -18, 9, -1]) / 6,
    array([24, -96, 72, -16, 1]) / 24,
    array([120, -600, 600, -200, 25, -1]) / 120,
    array([720, -4320, 5400, -2400, 450, -36, 1]) / 720,
]

legcoefficients = [
    array([1]),
    array([0, 1]),
    array([-1, 0, 3]) / 2,
    array([0, -3, 0, 5]) / 2,
    array([3, 0, -30, 0, 35]) / 8,
    array([0, 15, 0, -70, 0, 63]) / 8,
    array([-5, 0, 105, 0, -315, 0, 231]) / 16,
    array([0, -35, 0, 315, 0, -693, 0, 429]) / 16,
    array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
    array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
]

polycoefficients = [
    array([1]),
    array([0, 1]),
    array([-1, 0, 2]),
    array([0, -3, 0, 4]),
    array([1, 0, -8, 0, 8]),
    array([0, 5, 0, -20, 0, 16]),
    array([-1, 0, 18, 0, -48, 0, 32]),
    array([0, -7, 0, 56, 0, -112, 0, 64]),
    array([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    array([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]


def test__c_series_to_z_series():
    for i in range(5):
        assert_array_almost_equal(
            _c_series_to_z_series(
                array([2] + [1] * i, dtype=numpy.float64),
            ),
            array([0.5] * i + [2] + [0.5] * i, dtype=numpy.float64),
        )


def test__fit():
    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            array([1]),
            array([1]),
            degree=-1,
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([[1]]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1]),
            array([[[1]]]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1, 2]),
            array([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1]),
            array([1, 2]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1]),
            array([1]),
            degree=0,
            weight=[[1]],
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1]),
            array([1]),
            degree=0,
            weight=array([1, 1]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            array([1]),
            array([1]),
            degree=(-1,),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            array([1]),
            array([1]),
            degree=(2, -1, 6),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1]),
            array([1]),
            degree=(),
        )


def test__get_domain():
    assert_array_almost_equal(
        _get_domain(
            array([1, 10, 3, -1]),
        ),
        array([-1, 10]),
    )

    assert_array_almost_equal(
        _get_domain(
            array([1 + 1j, 1 - 1j, 0, 2]),
        ),
        array([-1j, 2 + 1j]),
    )


def test__map_domain():
    assert_array_almost_equal(
        _map_domain(
            array([0, 4]),
            array([0, 4]),
            array([1, 3]),
        ),
        array([1, 3]),
    )

    assert_array_almost_equal(
        _map_domain(
            array([0 - 1j, 2 + 1j]),
            array([0 - 1j, 2 + 1j]),
            array([-2, 2]),
        ),
        array([-2, 2]),
    )

    assert_array_almost_equal(
        _map_domain(
            array([[0, 4], [0, 4]]),
            array([0, 4]),
            array([1, 3]),
        ),
        array([[1, 3], [1, 3]]),
    )


def test__map_parameters():
    assert_array_almost_equal(
        _map_parameters(
            array([0, 4]),
            array([1, 3]),
        ),
        array([1, 0.5]),
    )

    assert_array_almost_equal(
        _map_parameters(
            array([+0 - 1j, +2 + 1j]),
            array([-2 + 0j, +2 + 0j]),
        ),
        array([-1 + 1j, +1 - 1j]),
    )


def test__pow():
    with pytest.raises(ValueError):
        _pow(
            (),
            array([1, 2, 3]),
            5,
            4,
        )


def test__trim_coefficients():
    with pytest.raises(ValueError):
        _trim_coefficients(
            array([2, -1, 1, 0]),
            tol=-1,
        )

    assert_array_almost_equal(
        _trim_coefficients(
            array([2, -1, 1, 0]),
        ),
        array([2, -1, 1, 0])[:-1],
    )

    assert_array_almost_equal(
        _trim_coefficients(
            array([2, -1, 1, 0]),
            tol=1,
        ),
        array([2, -1, 1, 0])[:-3],
    )

    assert_array_almost_equal(
        _trim_coefficients(
            array(array([2, -1, 1, 0])),
            tol=2,
        ),
        array([0]),
    )


def test__trim_sequence():
    for _ in range(5):
        assert_array_almost_equal(
            _trim_sequence(
                array([1] + [0] * 5),
            ),
            array([1]),
        )


def test__vandermonde():
    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (1, 2, 3),
            array([90]),
        )

    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (),
            array([90.65]),
        )

    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (),
            array([]),
        )


def test__z_series_to_c_series():
    for i in range(5):
        assert_array_almost_equal(
            _z_series_to_c_series(
                array([0.5] * i + [2] + [0.5] * i, dtype=numpy.float64),
            ),
            array([2] + [1] * i, dtype=numpy.float64),
        )


def test_cheb2poly():
    for i in range(10):
        assert_array_almost_equal(
            cheb2poly(
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
                chebtrim(
                    chebadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebcompanion():
    with pytest.raises(ValueError):
        chebcompanion(array([]))

    with pytest.raises(ValueError):
        chebcompanion(array([1]))

    for i in range(1, 5):
        assert chebcompanion(array([0] * i + [1])).shape == (i, i)

    assert chebcompanion(array([1, 2]))[0, 0] == -0.5


def test_chebder():
    with pytest.raises(TypeError):
        chebder(array([0]), 0.5)

    with pytest.raises(ValueError):
        chebder(array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            chebtrim(
                chebder(
                    array([0] * i + [1]),
                    order=0,
                ),
                tol=0.000001,
            ),
            chebtrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                chebtrim(
                    chebder(
                        chebint(
                            array([0] * i + [1]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                chebtrim(
                    chebder(
                        chebint(
                            array([0] * i + [1]),
                            order=j,
                            scl=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        chebder(
            c2d,
            axis=0,
        ),
        vstack([chebder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        chebder(
            c2d,
            axis=1,
        ),
        vstack([chebder(c) for c in c2d]),
    )


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = chebdiv(
                chebadd(
                    array([0] * i + [1]),
                    array([0] * j + [1]),
                ),
                array([0] * i + [1]),
            )

            assert_array_almost_equal(
                chebtrim(
                    chebadd(
                        chebmul(
                            quotient,
                            array([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    chebadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebdomain():
    assert_array_almost_equal(
        chebdomain,
        array([-1, 1]),
    )


def test_chebfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=(0, 1, 2, 3),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=(0, 1, 2, 3, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=(2, 3, 4, 1, 0),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebfit(
            input,
            array([other, other]).T,
            degree=3,
        ),
        array(
            [
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        chebfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
        ),
        array(
            [
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight = weight.at[1::2].set(1)

    assert_array_almost_equal(
        chebfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        chebfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        chebfit(
            input,
            other,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        chebfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        chebfit(
            input,
            array([other, other]).T,
            degree=3,
            weight=weight,
        ),
        array(
            [
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        chebfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        array(
            [
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                chebfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ],
        ).T,
    )

    assert_array_almost_equal(
        chebfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=1,
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        chebfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        array([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=(0, 2, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        chebfit(
            input,
            other,
            degree=4,
        ),
        chebfit(
            input,
            other,
            degree=(0, 2, 4),
        ),
    )


def test_chebfromroots():
    assert_array_almost_equal(
        chebtrim(
            chebfromroots(
                array([]),
            ),
            tol=0.000001,
        ),
        array([1]),
    )
    for i in range(1, 5):
        input = chebfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2]))

        input = input * 2 ** (i - 1)

        assert_array_almost_equal(
            chebtrim(
                input,
                tol=0.000001,
            ),
            chebtrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )


def test_chebgauss():
    point, weight = chebgauss(100)

    t = chebvander(point, 99)

    u = dot(transpose(t) * weight, t)

    v = 1 / sqrt(u.diagonal())

    assert_array_almost_equal(
        v[:, None] * u * v,
        eye(100),
    )

    assert_array_almost_equal(
        sum(weight),
        math.pi,
    )


def test_chebgrid2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x

    y1, y2, y3 = polyval(
        x,
        array([1.0, 2.0, 3.0]),
    )

    assert_array_almost_equal(
        chebgrid2d(
            x1,
            x2,
            einsum(
                "i,j->ij",
                array([2.5, 2.0, 1.5]),
                array([2.5, 2.0, 1.5]),
            ),
        ),
        einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    z = ones([2, 3])

    res = chebgrid2d(
        z,
        z,
        einsum(
            "i,j->ij",
            array([2.5, 2.0, 1.5]),
            array([2.5, 2.0, 1.5]),
        ),
    )

    assert res.shape == (2, 3) * 2


def test_chebgrid3d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(
        x,
        array([1.0, 2.0, 3.0]),
    )

    assert_array_almost_equal(
        chebgrid3d(
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

    output = chebgrid3d(
        ones([2, 3]),
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j,k->ijk",
            array([2.5, 2.0, 1.5]),
            array([2.5, 2.0, 1.5]),
            array([2.5, 2.0, 1.5]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_chebint():
    pytest.raises(TypeError, chebint, array([0]), 0.5)
    pytest.raises(ValueError, chebint, array([0]), -1)
    pytest.raises(ValueError, chebint, array([0]), 1, [0, 0])
    pytest.raises(ValueError, chebint, array([0]), lbnd=[0])
    pytest.raises(ValueError, chebint, array([0]), scl=[0])
    pytest.raises(TypeError, chebint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]

        assert_array_almost_equal(
            chebtrim(
                chebint(
                    array([0]),
                    order=i,
                    k=k,
                ),
                tol=0.000001,
            ),
            [0, 1],
        )

    for i in range(5):
        assert_array_almost_equal(
            chebtrim(
                cheb2poly(
                    chebint(poly2cheb(array([0] * i + [1])), order=1, k=[i]),
                ),
                tol=0.000001,
            ),
            chebtrim(
                array([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_array_almost_equal(
            chebval(
                array([-1]),
                chebint(
                    poly2cheb(array([0] * i + [1])),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_array_almost_equal(
            chebtrim(
                cheb2poly(
                    chebint(
                        poly2cheb(array([0] * i + [1])),
                        order=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tol=0.000001,
            ),
            chebtrim(
                array([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]

            for _ in range(j):
                target = chebint(target, order=1)

            assert_array_almost_equal(
                chebtrim(
                    chebint(pol, order=j),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]

            for k in range(j):
                target = chebint(target, order=1, k=[k])

            assert_array_almost_equal(
                chebtrim(
                    chebint(
                        pol,
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]

            for k in range(j):
                target = chebint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )

            assert_array_almost_equal(
                chebtrim(
                    chebint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = chebint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                chebtrim(
                    chebint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([chebint(c) for c in c2d.T]).T
    assert_array_almost_equal(
        chebint(c2d, axis=0),
        target,
    )

    target = vstack([chebint(c) for c in c2d])
    assert_array_almost_equal(
        chebint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = vstack([chebint(c, k=3) for c in c2d])
    assert_array_almost_equal(
        chebint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )


def test_chebinterpolate():
    def f(x):
        return x * (x - 1) * (x - 2)

    pytest.raises(ValueError, chebinterpolate, f, -1)

    for deg in range(1, 5):
        assert chebinterpolate(f, deg).shape == (deg + 1,)

    def powx(x, p):
        return x**p

    x = linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = chebinterpolate(powx, deg, (p,))
            assert_array_almost_equal(
                chebval(x, c),
                powx(x, p),
            )


def test_chebline():
    assert_array_almost_equal(chebline(3, 4), array([3, 4]))


def test_chebmul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[abs(i + j)].set(target[abs(i + j)] + 0.5)
            target = target.at[abs(i - j)].set(target[abs(i - j)] + 0.5)

            assert_array_almost_equal(
                chebtrim(
                    chebmul(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebmulx():
    assert_array_almost_equal(
        chebtrim(
            chebmulx([0]),
            tol=0.000001,
        ),
        [0],
    )

    assert_array_almost_equal(
        chebtrim(
            chebmulx([1]),
            tol=0.000001,
        ),
        [0, 1],
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            chebtrim(
                chebmulx(array([0] * i + [1])),
                tol=0.000001,
            ),
            [0] * (i - 1) + [0.5, 0, 0.5],
        )


def test_chebone():
    assert_array_almost_equal(
        chebone,
        array([1]),
    )


def test_chebpow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                chebtrim(
                    chebpow(arange(i + 1), j),
                    tol=0.000001,
                ),
                chebtrim(
                    functools.reduce(
                        chebmul,
                        [(arange(i + 1))] * j,
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebpts1():
    with pytest.raises(ValueError):
        chebpts1(1.5)

    with pytest.raises(ValueError):
        chebpts1(0)

    assert_array_almost_equal(chebpts1(1), [0])

    assert_array_almost_equal(chebpts1(2), [-0.70710678118654746, 0.70710678118654746])

    assert_array_almost_equal(
        chebpts1(3), [-0.86602540378443871, 0, 0.86602540378443871]
    )

    assert_array_almost_equal(
        chebpts1(4),
        [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325],
    )


def test_chebpts2():
    with pytest.raises(ValueError):
        chebpts2(1.5)

    with pytest.raises(ValueError):
        chebpts2(1)

    assert_array_almost_equal(chebpts2(2), [-1, 1])

    assert_array_almost_equal(chebpts2(3), array([-1, 0, 1]))

    assert_array_almost_equal(chebpts2(4), [-1, -0.5, 0.5, 1])

    assert_array_almost_equal(
        chebpts2(5), [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
    )


def test_chebroots():
    assert_array_almost_equal(
        chebroots([1]),
        array([]),
    )

    assert_array_almost_equal(
        chebroots(array([1, 2])),
        array([-0.5]),
    )

    for i in range(2, 5):
        assert_array_almost_equal(
            chebtrim(
                chebroots(
                    chebfromroots(
                        linspace(-1, 1, i),
                    )
                ),
                tol=0.000001,
            ),
            chebtrim(
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
                chebtrim(
                    chebsub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebtrim():
    pytest.raises(ValueError, chebtrim, array([2, -1, 1, 0]), -1)

    assert_array_almost_equal(chebtrim(array([2, -1, 1, 0])), array([2, -1, 1, 0])[:-1])

    assert_array_almost_equal(
        chebtrim(array([2, -1, 1, 0]), 1), array([2, -1, 1, 0])[:-3]
    )

    assert_array_almost_equal(
        chebtrim(array([2, -1, 1, 0]), 2),
        array([0]),
    )


def test_chebval():
    assert_array_almost_equal(chebval(array([]), [1]).size, 0)

    x = linspace(-1, 1, 50)
    y = [polyval(x, c) for c in chebcoefficients]
    for i in range(10):
        target = y[i]
        res = chebval(x, array([0] * i + [1]))
        assert_array_almost_equal(
            res,
            target,
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(chebval(x, [1]).shape, dims)
        assert_array_almost_equal(chebval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(chebval(x, array([1, 0, 0])).shape, dims)


def test_chebval2d():
    c1d = array([2.5, 2.0, 1.5])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        chebval2d,
        x1,
        x2[:2],
        c2d,
    )

    target = y1 * y2
    res = chebval2d(
        x1,
        x2,
        c2d,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = chebval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_chebval3d():
    c1d = array([2.5, 2.0, 1.5])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, chebval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    res = chebval3d(x1, x2, x3, c3d)
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = chebval3d(z, z, z, c3d)
    assert res.shape == (2, 3)


def test_chebvander():
    x = arange(3)
    v = chebvander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], chebval(x, coef))

    x = array([[1, 2], [3, 4], [5, 6]])
    v = chebvander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], chebval(x, coef))


def test_chebvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    van = chebvander2d(x1, x2, (1, 2))

    assert_array_almost_equal(
        dot(
            van,
            ravel(c),
        ),
        chebval2d(
            x1,
            x2,
            c,
        ),
    )

    van = chebvander2d([x1], [x2], (1, 2))
    assert van.shape == (1, 5, 6)


def test_chebvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(chebvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        chebval3d(x1, x2, x3, c),
    )

    van = chebvander3d([x1], [x2], [x3], (1, 2, 3))
    assert van.shape == (1, 5, 24)


def test_chebweight():
    x = linspace(-1, 1, 11)[1:-1]
    assert_array_almost_equal(
        chebweight(x),
        1.0 / (sqrt(1 + x) * sqrt(1 - x)),
    )


def test_chebx():
    assert_array_almost_equal(
        chebx,
        array([0, 1]),
    )


def test_chebzero():
    assert_array_almost_equal(chebzero, array([0]))


def test_herm2poly():
    for i in range(10):
        assert_array_almost_equal(
            herm2poly(array([0] * i + [1])),
            hermcoefficients[i],
        )


def test_hermadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                hermtrim(
                    hermadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermcompanion():
    with pytest.raises(ValueError):
        hermcompanion(array([]))
    with pytest.raises(ValueError):
        hermcompanion([1])

    for i in range(1, 5):
        assert hermcompanion(array([0] * i + [1])).shape == (i, i)

    assert hermcompanion(array([1, 2]))[0, 0] == -0.25


def test_hermder():
    with pytest.raises(TypeError):
        hermder(array([0]), 0.5)

    with pytest.raises(ValueError):
        hermder(array([0]), -1)

    for i in range(5):
        target = array([0] * i + [1])
        res = hermder(target, order=0)
        assert_array_almost_equal(
            hermtrim(
                res,
                tol=0.000001,
            ),
            hermtrim(
                target,
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = array([0] * i + [1])
            res = hermder(hermint(target, order=j), order=j)
            assert_array_almost_equal(
                hermtrim(
                    res,
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = array([0] * i + [1])
            res = hermder(hermint(target, order=j, scl=2), order=j, scl=0.5)
            assert_array_almost_equal(
                hermtrim(
                    res,
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([hermder(c) for c in c2d.T]).T
    res = hermder(c2d, axis=0)
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([hermder(c) for c in c2d])
    res = hermder(
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
            quotient, remainder = hermdiv(
                hermadd(
                    array([0.0] * i + [1.0]),
                    array([0.0] * j + [1.0]),
                ),
                array([0.0] * i + [1.0]),
            )

            assert_array_almost_equal(
                hermtrim(
                    hermadd(
                        hermmul(
                            quotient,
                            array([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    hermadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermdomain():
    assert_array_almost_equal(hermdomain, array([-1, 1]))


def test_herme2poly():
    for i in range(10):
        assert_array_almost_equal(
            herme2poly(array([0] * i + [1])),
            hermecoefficients[i],
        )


def test_hermeadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                hermetrim(
                    hermeadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermecompanion():
    with pytest.raises(ValueError):
        hermecompanion(array([]))

    with pytest.raises(ValueError):
        hermecompanion([1])

    for i in range(1, 5):
        coef = array([0] * i + [1])
        assert hermecompanion(coef).shape == (i, i)

    assert hermecompanion(array([1, 2]))[0, 0] == -0.5


def test_hermeder():
    pytest.raises(TypeError, hermeder, array([0]), 0.5)
    pytest.raises(ValueError, hermeder, array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            hermetrim(
                hermeder(array([0] * i + [1]), order=0),
                tol=0.000001,
            ),
            hermetrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                hermetrim(
                    hermeder(
                        hermeint(array([0] * i + [1]), order=j),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                hermetrim(
                    hermeder(
                        hermeint(array([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        hermeder(c2d, axis=0),
        vstack([hermeder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        hermeder(
            c2d,
            axis=1,
        ),
        vstack([hermeder(c) for c in c2d]),
    )


def test_hermediv():
    for i in range(5):
        for j in range(5):
            ci = array([0] * i + [1])
            cj = array([0] * j + [1])
            quotient, remainder = hermediv(hermeadd(ci, cj), ci)
            assert_array_almost_equal(
                hermetrim(
                    hermeadd(hermemul(quotient, ci), remainder),
                    tol=0.000001,
                ),
                hermetrim(
                    hermeadd(ci, cj),
                    tol=0.000001,
                ),
            )


def test_hermedomain():
    assert_array_almost_equal(hermedomain, [-1, 1])


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=(0, 1, 2, 3),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=(0, 1, 2, 3, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=(2, 3, 4, 1, 0),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermefit(
            input,
            array([other, other]).T,
            degree=3,
        ),
        array(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        hermefit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
        ),
        array(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight = weight.at[1::2].set(1)

    assert_array_almost_equal(
        hermefit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        hermefit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        hermefit(
            input,
            other,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        hermefit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        hermefit(
            input,
            array([other, other]).T,
            degree=3,
            weight=weight,
        ),
        array(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        hermefit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        array(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        hermefit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=1,
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        hermefit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        array([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=(0, 2, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermefit(
            input,
            other,
            degree=4,
        ),
        hermefit(
            input,
            other,
            degree=(0, 2, 4),
        ),
    )


def test_hermefromroots():
    res = hermefromroots(array([]))
    assert_array_almost_equal(
        hermetrim(
            res,
            tol=0.000001,
        ),
        array([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = hermefromroots(roots)
        assert len(pol) == i + 1
        assert_array_almost_equal(herme2poly(pol)[-1], 1)
        assert_array_almost_equal(
            hermeval(roots, pol),
            0,
        )


def test_hermegauss():
    x, w = hermegauss(100)

    v = hermevander(x, 99)
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
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j->ij", y1, y2)
    res = hermegrid2d(
        x1,
        x2,
        c2d,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    z = ones([2, 3])
    res = hermegrid2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3) * 2


def test_hermegrid3d():
    c1d = array([4.0, 2.0, 3.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    assert_array_almost_equal(
        hermegrid3d(x1, x2, x3, c3d),
        einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = ones([2, 3])
    res = hermegrid3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3


def test_hermeint():
    pytest.raises(TypeError, hermeint, array([0]), 0.5)
    pytest.raises(ValueError, hermeint, array([0]), -1)
    pytest.raises(ValueError, hermeint, array([0]), 1, [0, 0])
    pytest.raises(ValueError, hermeint, array([0]), lbnd=[0])
    pytest.raises(ValueError, hermeint, array([0]), scl=[0])
    pytest.raises(TypeError, hermeint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = hermeint(array([0]), order=i, k=k)
        assert_array_almost_equal(
            hermetrim(
                res,
                tol=0.000001,
            ),
            array([0, 1]),
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [1 / scl]
        hermepol = poly2herme(pol)
        res = herme2poly(hermeint(hermepol, order=1, k=[i]))
        assert_array_almost_equal(
            hermetrim(
                res,
                tol=0.000001,
            ),
            hermetrim(
                target,
                tol=0.000001,
            ),
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        hermepol = poly2herme(pol)
        assert_array_almost_equal(
            hermeval(
                array([-1]),
                hermeint(
                    hermepol,
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [2 / scl]
        hermepol = poly2herme(pol)
        res = herme2poly(hermeint(hermepol, order=1, k=[i], scl=2))
        assert_array_almost_equal(
            hermetrim(
                res,
                tol=0.000001,
            ),
            hermetrim(
                target,
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = hermeint(target, order=1)
            res = hermeint(pol, order=j)
            assert_array_almost_equal(
                hermetrim(
                    res,
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermeint(target, order=1, k=[k])

            assert_array_almost_equal(
                hermetrim(
                    hermeint(pol, order=j, k=list(range(j))),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermeint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )
            assert_array_almost_equal(
                hermetrim(
                    hermeint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermeint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                hermetrim(
                    hermeint(pol, order=j, k=list(range(j)), scl=2), tol=0.000001
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        hermeint(c2d, axis=0),
        vstack([hermeint(c) for c in c2d.T]).T,
    )

    target = vstack([hermeint(c) for c in c2d])
    res = hermeint(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([hermeint(c, k=3) for c in c2d])
    res = hermeint(
        c2d,
        k=3,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )


def test_hermeline():
    assert_array_almost_equal(hermeline(3, 4), array([3, 4]))


def test_hermemul():
    x = linspace(-3, 3, 100)
    for i in range(5):
        pol1 = array([0] * i + [1])
        val1 = hermeval(x, pol1)
        for j in range(5):
            pol2 = array([0] * j + [1])
            val2 = hermeval(x, pol2)
            pol3 = hermemul(pol1, pol2)
            val3 = hermeval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_array_almost_equal(
                val3,
                val1 * val2,
            )


def test_hermemulx():
    assert_array_almost_equal(
        hermetrim(
            hermemulx([0]),
            tol=0.000001,
        ),
        [0],
    )
    assert_array_almost_equal(
        hermetrim(
            hermemulx([1]),
            tol=0.000001,
        ),
        [0, 1],
    )
    for i in range(1, 5):
        assert_array_almost_equal(
            hermetrim(
                hermemulx(array([0] * i + [1])),
                tol=0.000001,
            ),
            [0] * (i - 1) + [i, 0, 1],
        )


def test_hermeone():
    assert_array_almost_equal(
        hermeone,
        array([1]),
    )


def test_hermepow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                hermetrim(
                    hermepow(
                        arange(i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    functools.reduce(
                        hermemul,
                        array([arange(i + 1)] * j),
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermeroots():
    assert_array_almost_equal(hermeroots([1]), array([]))

    assert_array_almost_equal(hermeroots([1, 1]), [-1])

    for i in range(2, 5):
        assert_array_almost_equal(
            hermetrim(
                hermeroots(
                    hermefromroots(
                        linspace(-1, 1, i),
                    )
                ),
                tol=0.000001,
            ),
            hermetrim(
                linspace(-1, 1, i),
                tol=0.000001,
            ),
        )


def test_hermesub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                hermetrim(
                    hermesub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermetrim():
    coef = array([2, -1, 1, 0])

    pytest.raises(ValueError, hermetrim, coef, -1)

    assert_array_almost_equal(hermetrim(coef), coef[:-1])
    assert_array_almost_equal(hermetrim(coef, 1), coef[:-3])
    assert_array_almost_equal(hermetrim(coef, 2), [0])


def test_hermeval():
    assert_array_almost_equal(hermeval(array([]), [1]).size, 0)

    x = linspace(-1, 1, 50)
    y = [polyval(x, c) for c in hermecoefficients]
    for i in range(10):
        assert_array_almost_equal(hermeval(x, array([0] * i + [1])), y[i])

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(hermeval(x, [1]).shape, dims)
        assert_array_almost_equal(hermeval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(hermeval(x, array([1, 0, 0])).shape, dims)


def test_hermeval2d():
    c1d = array([4.0, 2.0, 3.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    pytest.raises(
        ValueError,
        hermeval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_array_almost_equal(
        hermeval2d(
            x1,
            x2,
            c2d,
        ),
        y1 * y2,
    )

    z = ones([2, 3])
    res = hermeval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_hermeval3d():
    c1d = array([4.0, 2.0, 3.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, hermeval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    res = hermeval3d(x1, x2, x3, c3d)
    assert_array_almost_equal(res, target)

    z = ones([2, 3])
    res = hermeval3d(z, z, z, c3d)
    assert res.shape == (2, 3)


def test_hermevander():
    x = arange(3)
    v = hermevander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], hermeval(x, coef))

    x = array([[1, 2], [3, 4], [5, 6]])
    v = hermevander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], hermeval(x, coef))


def test_hermevander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(hermevander2d(x1, x2, (1, 2)), c.ravel()),
        hermeval2d(x1, x2, c),
    )

    van = hermevander2d([x1], [x2], (1, 2))
    assert van.shape == (1, 5, 6)


def test_hermevander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    van = hermevander3d(x1, x2, x3, (1, 2, 3))
    assert_array_almost_equal(dot(van, c.ravel()), hermeval3d(x1, x2, x3, c))

    van = hermevander3d([x1], [x2], [x3], (1, 2, 3))
    assert van.shape == (1, 5, 24)


def test_hermeweight():
    x = linspace(-5, 5, 11)
    target = exp(-0.5 * x**2)
    res = hermeweight(x)
    assert_array_almost_equal(
        res,
        target,
    )


def test_hermex():
    assert_array_almost_equal(
        hermex,
        array([0, 1]),
    )


def test_hermezero():
    assert_array_almost_equal(
        hermezero,
        array([0]),
    )


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=(0, 1, 2, 3),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=(0, 1, 2, 3, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=(2, 3, 4, 1, 0),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermfit(
            input,
            array([other, other]).T,
            degree=3,
        ),
        array(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ],
        ).T,
    )

    assert_array_almost_equal(
        hermfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
        ),
        array(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight = weight.at[1::2].set(1)

    assert_array_almost_equal(
        hermfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        hermfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        hermfit(
            input,
            other,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        hermfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        hermfit(
            input,
            array([other, other]).T,
            degree=3,
            weight=weight,
        ),
        array(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        hermfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        array(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        hermfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=1,
        ),
        array([0, 0.5]),
    )

    assert_array_almost_equal(
        hermfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        array([0, 0.5]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=(0, 2, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        hermfit(
            input,
            other,
            degree=4,
        ),
        hermfit(
            input,
            other,
            degree=(0, 2, 4),
        ),
    )


def test_hermfromroots():
    res = hermfromroots(array([]))
    assert_array_almost_equal(
        hermtrim(
            res,
            tol=0.000001,
        ),
        array([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = hermfromroots(roots)
        res = hermval(roots, pol)
        target = 0
        assert len(pol) == i + 1
        assert_array_almost_equal(herm2poly(pol)[-1], 1)
        assert_array_almost_equal(
            res,
            target,
        )


def test_hermgauss():
    x, w = hermgauss(100)

    v = hermvander(x, 99)
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
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    target = einsum("i,j->ij", y1, y2)
    assert_array_almost_equal(
        hermgrid2d(
            x1,
            x2,
            c2d,
        ),
        target,
    )

    z = ones([2, 3])
    assert (
        hermgrid2d(
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
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    assert_array_almost_equal(
        hermgrid3d(
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

    assert hermgrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_hermint():
    pytest.raises(TypeError, hermint, array([0]), 0.5)
    pytest.raises(ValueError, hermint, array([0]), -1)
    pytest.raises(
        ValueError,
        hermint,
        array([0]),
        1,
        array([0, 0]),
    )
    pytest.raises(ValueError, hermint, array([0]), lbnd=[0])
    pytest.raises(ValueError, hermint, array([0]), scl=[0])
    pytest.raises(TypeError, hermint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]

        assert_array_almost_equal(
            hermtrim(
                hermint(
                    array([0]),
                    order=i,
                    k=k,
                ),
                tol=0.000001,
            ),
            [0, 0.5],
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        hermpol = poly2herm(pol)
        assert_array_almost_equal(
            hermtrim(
                herm2poly(
                    hermint(
                        hermpol,
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            hermtrim(
                [i] + [0] * i + [1 / scl],
                tol=0.000001,
            ),
        )

    for i in range(5):
        pol = array([0] * i + [1])
        hermpol = poly2herm(pol)
        assert_array_almost_equal(
            hermval(
                array(-1),
                hermint(
                    hermpol,
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        hermpol = poly2herm(pol)
        assert_array_almost_equal(
            hermtrim(
                herm2poly(
                    hermint(hermpol, order=1, k=[i], scl=2),
                ),
                tol=0.000001,
            ),
            hermtrim(
                [i] + [0] * i + [2 / scl],
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = hermint(
                    target,
                    order=1,
                )

            assert_array_almost_equal(
                hermtrim(
                    hermint(
                        pol,
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermint(target, order=1, k=[k])
            assert_array_almost_equal(
                hermtrim(
                    hermint(pol, order=j, k=list(range(j))),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )

            assert_array_almost_equal(
                hermtrim(
                    hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermint(
                    target,
                    order=1,
                    k=[k],
                    scl=2,
                )

            assert_array_almost_equal(
                hermtrim(
                    hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scl=2,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([hermint(c) for c in c2d.T]).T
    assert_array_almost_equal(hermint(c2d, axis=0), target)

    target = vstack([hermint(c) for c in c2d])
    assert_array_almost_equal(
        hermint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = vstack([hermint(c, k=3) for c in c2d])

    assert_array_almost_equal(
        hermint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )


def test_hermline():
    assert_array_almost_equal(
        hermline(3, 4),
        [3, 2],
    )


def test_hermmul():
    x = linspace(-3, 3, 100)

    for i in range(5):
        pol1 = array([0.0] * i + [1.0])

        val1 = hermval(
            x,
            pol1,
        )

        for j in range(5):
            pol2 = array([0.0] * j + [1.0])
            val2 = hermval(
                x,
                pol2,
            )
            pol3 = hermmul(
                pol1,
                pol2,
            )
            val3 = hermval(
                x,
                pol3,
            )

            assert len(hermtrim(pol3, tol=0.000001)) == i + j + 1

            assert_array_almost_equal(
                val3,
                val1 * val2,
            )


def test_hermmulx():
    assert_array_almost_equal(
        hermtrim(
            hermmulx([0.0]),
            tol=0.000001,
        ),
        [0.0],
    )
    assert_array_almost_equal(
        hermmulx([1.0]),
        [0.0, 0.5],
    )
    for i in range(1, 5):
        assert_array_almost_equal(
            hermmulx(
                array([0.0] * i + [1.0]),
            ),
            [0.0] * (i - 1) + [i, 0.0, 0.5],
        )


def test_hermone():
    assert_array_almost_equal(hermone, array([1]))


def test_hermpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)

            assert_array_almost_equal(
                hermtrim(
                    hermpow(
                        c,
                        j,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    functools.reduce(
                        hermmul,
                        array([c] * j),
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermroots():
    assert_array_almost_equal(
        hermroots(
            array([1]),
        ),
        array([]),
    )

    assert_array_almost_equal(
        hermroots(
            array([1, 1]),
        ),
        array([-0.5]),
    )

    for i in range(2, 5):
        input = linspace(-1, 1, i)

        assert_array_almost_equal(
            hermtrim(
                hermroots(
                    hermfromroots(
                        input,
                    ),
                ),
                tol=0.000001,
            ),
            hermtrim(
                input,
                tol=0.000001,
            ),
        )


def test_hermsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                hermtrim(
                    hermsub(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermtrim():
    coef = array([2, -1, 1, 0])

    pytest.raises(ValueError, hermtrim, coef, -1)

    assert_array_almost_equal(hermtrim(coef), coef[:-1])
    assert_array_almost_equal(hermtrim(coef, 1), coef[:-3])
    assert_array_almost_equal(
        hermtrim(coef, 2),
        array([0]),
    )


def test_hermval():
    assert hermval(array([]), [1]).size == 0

    x = linspace(-1, 1, 50)
    y = [polyval(x, c) for c in hermcoefficients]

    for index in range(10):
        assert_array_almost_equal(
            hermval(
                x,
                [0] * index + [1],
            ),
            y[index],
        )

    for index in range(3):
        dims = (2,) * index
        x = zeros(dims)
        assert hermval(x, [1]).shape == dims
        assert hermval(x, array([1, 0])).shape == dims
        assert hermval(x, array([1, 0, 0])).shape == dims


def test_hermval2d():
    c1d = array([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        hermval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_array_almost_equal(
        hermval2d(
            x1,
            x2,
            c2d,
        ),
        y1 * y2,
    )

    z = ones([2, 3])
    res = hermval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_hermval3d():
    c1d = array([2.5, 1.0, 0.75])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, hermval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    assert_array_almost_equal(
        hermval3d(x1, x2, x3, c3d),
        target,
    )

    z = ones([2, 3])
    assert hermval3d(z, z, z, c3d).shape == (2, 3)


def test_hermvander():
    x = arange(3)
    v = hermvander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], hermval(x, coef))

    x = array([[1, 2], [3, 4], [5, 6]])
    v = hermvander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = array([0] * i + [1])
        assert_array_almost_equal(v[..., i], hermval(x, coef))


def test_hermvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(hermvander2d(x1, x2, (1, 2)), c.ravel()),
        hermval2d(x1, x2, c),
    )

    assert hermvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_hermvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(hermvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        hermval3d(x1, x2, x3, c),
    )

    assert hermvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_hermweight():
    assert_array_almost_equal(
        hermweight(linspace(-5, 5, 11)),
        exp(-(linspace(-5, 5, 11) ** 2)),
    )


def test_hermx():
    assert_array_almost_equal(hermx, array([0, 0.5]))


def test_hermzero():
    assert_array_almost_equal(hermzero, array([0]))


def test_lag2poly():
    for i in range(7):
        assert_array_almost_equal(lag2poly(array([0] * i + [1])), lagcoefficients[i])


def test_lagadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                lagtrim(
                    lagadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_lagcompanion():
    with pytest.raises(ValueError):
        lagcompanion(array([]))
    with pytest.raises(ValueError):
        lagcompanion([1])

    for i in range(1, 5):
        coef = array([0] * i + [1])
        assert lagcompanion(coef).shape == (i, i)

    assert lagcompanion(array([1, 2]))[0, 0] == 1.5


def test_lagder():
    pytest.raises(TypeError, lagder, array([0]), 0.5)
    pytest.raises(ValueError, lagder, array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            lagtrim(
                lagder(array([0] * i + [1]), order=0),
                tol=0.000001,
            ),
            lagtrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                lagtrim(
                    lagder(lagint(array([0] * i + [1]), order=j), order=j),
                    tol=0.000001,
                ),
                lagtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                lagtrim(
                    lagder(
                        lagint(array([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        lagder(c2d, axis=0),
        vstack([lagder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        lagder(
            c2d,
            axis=1,
        ),
        vstack([lagder(c) for c in c2d]),
    )


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = lagdiv(
                lagadd(
                    array([0] * i + [1]),
                    array([0] * j + [1]),
                ),
                array([0] * i + [1]),
            )

            assert_array_almost_equal(
                lagtrim(
                    lagadd(
                        lagmul(
                            quotient,
                            array([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    lagadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagdomain():
    assert_array_almost_equal(
        lagdomain,
        array([0, 1]),
    )


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    input = linspace(0, 2, 50)

    other = f(input)

    assert_array_almost_equal(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=(0, 1, 2, 3),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=(0, 1, 2, 3, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        lagfit(
            input,
            array([other, other]).T,
            degree=3,
        ),
        array(
            [
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        lagfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
        ),
        array(
            [
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight = weight.at[1::2].set(1)

    assert_array_almost_equal(
        lagfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        lagfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        lagfit(
            input,
            other,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        lagfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        lagfit(
            input,
            array([other, other]).T,
            degree=3,
            weight=weight,
        ),
        array(
            [
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ],
        ).T,
    )

    assert_array_almost_equal(
        lagfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        array(
            [
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                lagfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        lagfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=1,
        ),
        array([1, -1]),
    )

    assert_array_almost_equal(
        lagfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        array([1, -1]),
    )


def test_lagfromroots():
    res = lagfromroots(array([]))
    assert_array_almost_equal(
        lagtrim(
            res,
            tol=0.000001,
        ),
        array([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = lagfromroots(roots)
        res = lagval(roots, pol)
        target = 0
        assert len(pol) == i + 1
        assert_array_almost_equal(lag2poly(pol)[-1], 1)
        assert_array_almost_equal(res, target)


def test_laggauss():
    x, w = laggauss(100)

    v = lagvander(x, 99)
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
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    assert_array_almost_equal(
        laggrid2d(
            x1,
            x2,
            c2d,
        ),
        einsum("i,j->ij", y1, y2),
    )

    z = ones([2, 3])
    assert (
        laggrid2d(
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
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j,k->ijk", y1, y2, y3)
    assert_array_almost_equal(laggrid3d(x1, x2, x3, c3d), target, decimal=3)

    z = ones([2, 3])
    assert laggrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_lagint():
    pytest.raises(TypeError, lagint, array([0]), 0.5)
    pytest.raises(ValueError, lagint, array([0]), -1)
    pytest.raises(
        ValueError,
        lagint,
        array([0]),
        1,
        array([0, 0]),
    )
    pytest.raises(ValueError, lagint, array([0]), lbnd=[0])
    pytest.raises(ValueError, lagint, array([0]), scl=[0])
    pytest.raises(TypeError, lagint, array([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_array_almost_equal(
            lagtrim(
                lagint(array([0]), order=i, k=k),
                tol=0.000001,
            ),
            [1, -1],
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [1 / scl]
        res = lag2poly(lagint(poly2lag(pol), order=1, k=[i]))
        assert_array_almost_equal(
            lagtrim(
                res,
                tol=0.000001,
            ),
            lagtrim(
                target,
                tol=0.000001,
            ),
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        lagpol = poly2lag(pol)
        assert_array_almost_equal(
            lagval(
                array([-1]),
                lagint(
                    lagpol,
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = array([0] * i + [1])
        target = [i] + [0] * i + [2 / scl]
        lagpol = poly2lag(pol)
        assert_array_almost_equal(
            lagtrim(
                lag2poly(lagint(lagpol, order=1, k=[i], scl=2)),
                tol=0.000001,
            ),
            lagtrim(
                target,
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = lagint(target, order=1)
            assert_array_almost_equal(
                lagtrim(
                    lagint(pol, order=j),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = lagint(target, order=1, k=[k])
            assert_array_almost_equal(
                lagtrim(
                    lagint(pol, order=j, k=list(range(j))),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = lagint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )
            assert_array_almost_equal(
                lagtrim(
                    lagint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = lagint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                lagtrim(
                    lagint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([lagint(c) for c in c2d.T]).T
    assert_array_almost_equal(
        lagint(c2d, axis=0),
        target,
    )

    target = vstack([lagint(c) for c in c2d])
    res = lagint(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([lagint(c, k=3) for c in c2d])
    res = lagint(
        c2d,
        k=3,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )


def test_lagline():
    assert_array_almost_equal(lagline(3, 4), [7, -4])


def test_lagmul():
    x = linspace(-3, 3, 100)

    for i in range(5):
        pol1 = array([0] * i + [1])
        val1 = lagval(x, pol1)
        for j in range(5):
            pol2 = array([0] * j + [1])
            val2 = lagval(x, pol2)
            pol3 = lagtrim(lagmul(pol1, pol2))
            val3 = lagval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_array_almost_equal(val3, val1 * val2)


def test_lagmulx():
    assert_array_almost_equal(
        lagtrim(
            lagmulx(
                [0],
            ),
            tol=0.000001,
        ),
        array([0]),
    )

    assert_array_almost_equal(
        lagtrim(
            lagmulx(
                array([1]),
            ),
            tol=0.000001,
        ),
        array([1, -1]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            lagtrim(
                lagmulx(
                    array([0] * i + [1]),
                ),
                tol=0.000001,
            ),
            lagtrim(
                array([0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]),
                tol=0.000001,
            ),
        )


def test_lagone():
    assert_array_almost_equal(
        lagone,
        array([1]),
    )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)
            assert_array_almost_equal(
                lagtrim(
                    lagpow(c, j),
                    tol=0.000001,
                ),
                lagtrim(
                    functools.reduce(lagmul, [c] * j, array([1])),
                    tol=0.000001,
                ),
            )


def test_lagroots():
    assert_array_almost_equal(lagroots(array([1])), array([]))
    assert_array_almost_equal(
        lagroots(array([0, 1])),
        array([1]),
    )
    for i in range(2, 5):
        assert_array_almost_equal(
            lagtrim(
                lagroots(lagfromroots(linspace(0, 3, i))),
                tol=0.000001,
            ),
            lagtrim(
                linspace(0, 3, i),
                tol=0.000001,
            ),
        )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_array_almost_equal(
                lagtrim(
                    lagsub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_lagtrim():
    pytest.raises(ValueError, lagtrim, array([2, -1, 1, 0]), -1)

    assert_array_almost_equal(lagtrim(array([2, -1, 1, 0])), array([2, -1, 1, 0])[:-1])
    assert_array_almost_equal(
        lagtrim(array([2, -1, 1, 0]), 1), array([2, -1, 1, 0])[:-3]
    )
    assert_array_almost_equal(
        lagtrim(array([2, -1, 1, 0]), 2),
        array([0]),
    )


def test_lagval():
    assert_array_almost_equal(lagval(array([]), [1]).size, 0)

    x = linspace(-1, 1, 50)
    y = [polyval(x, c) for c in lagcoefficients]
    for i in range(7):
        assert_array_almost_equal(
            lagval(x, array([0] * i + [1])),
            y[i],
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(lagval(x, [1]).shape, dims)
        assert_array_almost_equal(lagval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(lagval(x, array([1, 0, 0])).shape, dims)


def test_lagval2d():
    c1d = array([9.0, -14.0, 6.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        lagval2d,
        x1,
        x2[:2],
        c2d,
    )

    target = y1 * y2
    assert_array_almost_equal(
        lagval2d(
            x1,
            x2,
            c2d,
        ),
        target,
        decimal=3,
    )

    z = ones([2, 3])
    assert lagval2d(
        z,
        z,
        c2d,
    ).shape == (2, 3)


def test_lagval3d():
    c1d = array([9.0, -14.0, 6.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, lagval3d, x1, x2, x3[:2], c3d)

    assert_array_almost_equal(
        lagval3d(
            x1,
            x2,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    assert lagval3d(ones([2, 3]), ones([2, 3]), ones([2, 3]), c3d).shape == (2, 3)


def test_lagvander():
    x = arange(3)

    v = lagvander(x, 3)

    assert v.shape == (3, 4)

    for i in range(4):
        assert_array_almost_equal(
            v[..., i],
            lagval(
                x,
                array([0] * i + [1]),
            ),
        )

    x = array([[1, 2], [3, 4], [5, 6]])

    v = lagvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_array_almost_equal(
            v[..., i],
            lagval(
                x,
                array([0] * i + [1]),
            ),
        )


def test_lagvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(lagvander2d(x1, x2, (1, 2)), c.ravel()),
        lagval2d(x1, x2, c),
    )

    assert lagvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_lagvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(lagvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        lagval3d(x1, x2, x3, c),
    )

    assert lagvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_lagweight():
    assert_array_almost_equal(
        lagweight(linspace(0, 10, 11)),
        exp(-linspace(0, 10, 11)),
    )


def test_lagx():
    assert_array_almost_equal(lagx, [1, -1])


def test_lagzero():
    assert_array_almost_equal(
        lagzero,
        array([0]),
    )


def test_leg2poly():
    for i in range(10):
        assert_array_almost_equal(
            leg2poly([0] * i + [1]),
            legcoefficients[i],
        )


def test_legadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_array_almost_equal(
                legtrim(
                    legadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_legcompanion():
    with pytest.raises(ValueError):
        legcompanion(array([]))

    with pytest.raises(ValueError):
        legcompanion(array([1]))

    for i in range(1, 5):
        coef = array([0] * i + [1])
        assert legcompanion(coef).shape == (i, i)

    assert legcompanion(array([1, 2]))[0, 0] == -0.5


def test_legder():
    pytest.raises(TypeError, legder, array([0]), 0.5)
    pytest.raises(ValueError, legder, array([0]), -1)

    for i in range(5):
        assert_array_almost_equal(
            legtrim(
                legder(array([0] * i + [1]), order=0),
                tol=0.000001,
            ),
            legtrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                legtrim(
                    legder(legint(array([0] * i + [1]), order=j), order=j), tol=0.000001
                ),
                legtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                legtrim(
                    legder(
                        legint(array([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([legder(c) for c in c2d.T]).T
    res = legder(c2d, axis=0)
    assert_array_almost_equal(
        res,
        target,
    )

    target = vstack([legder(c) for c in c2d])
    res = legder(
        c2d,
        axis=1,
    )
    assert_array_almost_equal(
        res,
        target,
    )

    c = (1, 2, 3, 4)
    assert_array_almost_equal(
        legder(c, 4),
        array([0]),
    )


def test_legdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = legdiv(
                legadd(
                    array([0] * i + [1]),
                    array([0] * j + [1]),
                ),
                array([0] * i + [1]),
            )

            assert_array_almost_equal(
                legtrim(
                    legadd(
                        legmul(
                            quotient,
                            array([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    legadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legdomain():
    assert_array_almost_equal(legdomain, [-1, 1])


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=(0, 1, 2, 3),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=(0, 1, 2, 3, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=(2, 3, 4, 1, 0),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legfit(
            input,
            array([other, other]).T,
            degree=3,
        ),
        array(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        legfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
        ),
        array(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight = weight.at[1::2].set(1)

    assert_array_almost_equal(
        legfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        legfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        legfit(
            input,
            other,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        legfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        legfit(
            input,
            array([other, other]).T,
            degree=3,
            weight=weight,
        ),
        array(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        legfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        array(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=(0, 1, 2, 3),
                    )
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        legfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=1,
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        legfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        array([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legval(
            input,
            legfit(
                input,
                other,
                degree=(0, 2, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        legfit(
            input,
            other,
            degree=4,
        ),
        legfit(
            input,
            other,
            degree=(0, 2, 4),
        ),
    )


def test_legfromroots():
    assert_array_almost_equal(
        legtrim(
            legfromroots(array([])),
            tol=0.000001,
        ),
        [1],
    )
    for i in range(1, 5):
        assert (
            legfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])).shape[-1] == i + 1
        )
        assert_array_almost_equal(
            leg2poly(legfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])))[-1],
            1,
        )
        assert_array_almost_equal(
            legval(
                cos(linspace(-math.pi, 0, 2 * i + 1)[1::2]),
                legfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])),
            ),
            0,
        )


def test_leggauss():
    x, w = leggauss(100)

    v = legvander(x, 99)
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
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    assert_array_almost_equal(
        leggrid2d(
            x1,
            x2,
            c2d,
        ),
        einsum("i,j->ij", y1, y2),
    )

    z = ones([2, 3])
    assert (
        leggrid2d(
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
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    assert_array_almost_equal(
        leggrid3d(
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

    assert leggrid3d(ones([2, 3]), ones([2, 3]), ones([2, 3]), c3d).shape == (2, 3) * 3


def test_legint():
    pytest.raises(TypeError, legint, array([0]), 0.5)
    pytest.raises(ValueError, legint, array([0]), -1)
    pytest.raises(
        ValueError,
        legint,
        array([0]),
        1,
        array([0, 0]),
    )
    pytest.raises(ValueError, legint, array([0]), lbnd=[0])
    pytest.raises(ValueError, legint, array([0]), scl=[0])
    pytest.raises(TypeError, legint, array([0]), axis=0.5)

    for i in range(2, 5):
        assert_array_almost_equal(
            legtrim(
                legint(array([0]), order=i, k=([0] * (i - 2) + [1])),
                tol=0.000001,
            ),
            [0, 1],
        )

    for i in range(5):
        assert_array_almost_equal(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(array([0] * i + [1])),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            legtrim(
                array([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_array_almost_equal(
            legval(
                array([-1]),
                legint(
                    poly2leg(array([0] * i + [1])),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_array_almost_equal(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(array([0] * i + [1])),
                        order=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tol=0.000001,
            ),
            legtrim(
                array([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = (array([0] * i + [1]))[:]
            for _ in range(j):
                target = legint(target, order=1)
            assert_array_almost_equal(
                legtrim(
                    legint(array([0] * i + [1]), order=j),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = legint(target, order=1, k=[k])
            assert_array_almost_equal(
                legtrim(
                    legint(pol, order=j, k=list(range(j))),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )
            assert_array_almost_equal(
                legtrim(
                    legint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = legint(target, order=1, k=[k], scl=2)
            assert_array_almost_equal(
                legtrim(
                    legint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        legint(c2d, axis=0),
        vstack([legint(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        legint(
            c2d,
            axis=1,
        ),
        vstack([legint(c) for c in c2d]),
    )

    assert_array_almost_equal(
        legint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([legint(c, k=3) for c in c2d]),
    )

    assert_array_almost_equal(legint((1, 2, 3), 0), (1, 2, 3))


def test_legline():
    assert_array_almost_equal(legline(3, 4), array([3, 4]))

    assert_array_almost_equal(
        legtrim(
            legline(3, 0),
            tol=0.000001,
        ),
        [3],
    )


def test_legmul():
    for i in range(5):
        pol1 = array([0] * i + [1])
        x = linspace(-1, 1, 100)
        val1 = legval(x, pol1)
        for j in range(5):
            pol2 = array([0] * j + [1])
            val2 = legval(x, pol2)
            pol3 = legmul(pol1, pol2)
            val3 = legval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_array_almost_equal(val3, val1 * val2)


def test_legmulx():
    assert_array_almost_equal(
        legtrim(
            legmulx(
                array([0]),
            ),
            tol=0.000001,
        ),
        array([0]),
    )

    assert_array_almost_equal(
        legtrim(
            legmulx(
                [1],
            ),
            tol=0.000001,
        ),
        [0, 1],
    )

    for index in range(1, 5):
        tmp = 2 * index + 1

        assert_array_almost_equal(
            legtrim(
                legmulx(
                    [0] * index + [1],
                ),
                tol=0.000001,
            ),
            [0] * (index - 1) + [index / tmp, 0, (index + 1) / tmp],
        )


def test_legone():
    assert_array_almost_equal(
        legone,
        array([1]),
    )


def test_legpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)

            assert_array_almost_equal(
                legtrim(
                    legpow(
                        c,
                        j,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    functools.reduce(
                        legmul,
                        [c] * j,
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legroots():
    assert_array_almost_equal(legroots([1]), array([]))
    assert_array_almost_equal(legroots(array([1, 2])), array([-0.5]))

    for index in range(2, 5):
        assert_array_almost_equal(
            legtrim(
                legroots(
                    legfromroots(
                        linspace(-1, 1, index),
                    ),
                ),
                tol=0.000001,
            ),
            legtrim(
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
                legtrim(
                    legsub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_legtrim():
    with pytest.raises(ValueError):
        legtrim(array([2, -1, 1, 0]), -1)

    assert_array_almost_equal(
        legtrim(array([2, -1, 1, 0])),
        array([2, -1, 1, 0])[:-1],
    )

    assert_array_almost_equal(
        legtrim(array([2, -1, 1, 0]), 1),
        array([2, -1, 1, 0])[:-3],
    )

    assert_array_almost_equal(
        legtrim(array([2, -1, 1, 0]), 2),
        array([0]),
    )


def test_legval():
    assert_array_almost_equal(legval(array([]), [1]).size, 0)

    x = linspace(-1, 1, 50)
    y = [polyval(x, c) for c in legcoefficients]
    for i in range(10):
        assert_array_almost_equal(legval(x, array([0] * i + [1])), y[i])

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_array_almost_equal(legval(x, array([1])).shape, dims)
        assert_array_almost_equal(legval(x, array([1, 0])).shape, dims)
        assert_array_almost_equal(legval(x, array([1, 0, 0])).shape, dims)


def test_legval2d():
    c1d = array([2.0, 2.0, 2.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        legval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_array_almost_equal(
        legval2d(
            x1,
            x2,
            c2d,
        ),
        y1 * y2,
    )

    z = ones([2, 3])
    assert legval2d(
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
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, legval3d, x1, x2, x3[:2], c3d)

    assert_array_almost_equal(
        legval3d(
            x1,
            x2,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    z = ones([2, 3])
    assert legval3d(z, z, z, c3d).shape == (2, 3)


def test_legvander():
    x = arange(3)
    v = legvander(x, 3)
    assert v.shape == (3, 4)

    for index in range(4):
        assert_array_almost_equal(
            v[..., index],
            legval(
                x,
                array([0] * index + [1]),
            ),
        )

    x = array([[1, 2], [3, 4], [5, 6]])
    v = legvander(x, 3)
    assert v.shape == (3, 2, 4)

    for index in range(4):
        assert_array_almost_equal(
            v[..., index],
            legval(
                x,
                array([0] * index + [1]),
            ),
        )

    pytest.raises(ValueError, legvander, (1, 2, 3), -1)


def test_legvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_array_almost_equal(
        dot(legvander2d(x1, x2, (1, 2)), c.ravel()),
        legval2d(x1, x2, c),
    )

    assert legvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_legvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_array_almost_equal(
        dot(legvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        legval3d(x1, x2, x3, c),
    )

    assert legvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_legweight():
    assert_array_almost_equal(
        legweight(linspace(-1, 1, 11)),
        1.0,
    )


def test_legx():
    assert_array_almost_equal(
        legx,
        array([0, 1]),
    )


def test_legzero():
    assert_array_almost_equal(
        legzero,
        array([0]),
    )


def test_poly2cheb():
    for i in range(10):
        assert_array_almost_equal(
            poly2cheb(
                chebcoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_poly2herm():
    for i in range(10):
        assert_array_almost_equal(
            hermtrim(
                poly2herm(
                    hermcoefficients[i],
                ),
                tol=0.000001,
            ),
            array([0] * i + [1]),
        )


def test_poly2herme():
    for i in range(10):
        assert_array_almost_equal(
            poly2herme(
                hermecoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_poly2lag():
    for i in range(7):
        assert_array_almost_equal(
            poly2lag(
                lagcoefficients[i],
            ),
            array([0] * i + [1]),
        )


def test_poly2leg():
    for i in range(10):
        assert_array_almost_equal(
            poly2leg(
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
                polytrim(
                    polyadd(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polycompanion():
    with pytest.raises(ValueError):
        polycompanion(array([]))

    with pytest.raises(ValueError):
        polycompanion(array([1]))

    for i in range(1, 5):
        output = polycompanion(
            array([0] * i + [1]),
        )

        assert output.shape == (i, i)

    output = polycompanion(
        array([1, 2]),
    )

    assert output[0, 0] == -0.5


def test_polyder():
    with pytest.raises(TypeError):
        polyder(array([0]), axis=0.5)

    for i in range(5):
        assert_array_almost_equal(
            polytrim(
                polyder(
                    array([0] * i + [1]),
                    order=0,
                ),
                tol=0.000001,
            ),
            polytrim(
                array([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                polytrim(
                    polyder(
                        polyint(
                            array([0] * i + [1]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_array_almost_equal(
                polytrim(
                    polyder(
                        polyint(
                            array([0] * i + [1]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    array([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_array_almost_equal(
        polyder(
            c2d,
            axis=0,
        ),
        vstack([polyder(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        polyder(
            c2d,
            axis=1,
        ),
        vstack([polyder(c) for c in c2d]),
    )


def test_polydiv():
    quotient, remainder = polydiv(
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

    quotient, remainder = polydiv(
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
            target = polyadd(
                array([0.0] * i + [1.0, 2.0]),
                array([0.0] * j + [1.0, 2.0]),
            )

            quotient, remainder = polydiv(
                target,
                array([0.0] * i + [1.0, 2.0]),
            )

            assert_array_almost_equal(
                polytrim(
                    polyadd(
                        polymul(
                            quotient,
                            array([0.0] * i + [1.0, 2.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polydomain():
    assert_array_almost_equal(
        polydomain,
        array([-1, 1]),
    )


def test_polyfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_array_almost_equal(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=(0, 1, 2, 3),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=(0, 1, 2, 3, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        polyfit(
            input,
            array([other, other]).T,
            degree=3,
        ),
        array(
            [
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        polyfit(
            input,
            array([other, other]).T,
            degree=(0, 1, 2, 3),
        ),
        array(
            [
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight = weight.at[1::2].set(1)

    assert_array_almost_equal(
        polyfit(
            input,
            other.at[0::2].set(0),
            degree=3,
            weight=weight,
        ),
        polyfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        polyfit(
            input,
            other.at[0::2].set(0),
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        polyfit(
            input,
            other,
            degree=(0, 1, 2, 3),
        ),
    )

    assert_array_almost_equal(
        polyfit(
            input,
            array([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
            degree=3,
            weight=weight,
        ),
        array(
            [
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        polyfit(
            input,
            array([(other.at[0::2].set(0)), (other.at[0::2].set(0))]).T,
            degree=(0, 1, 2, 3),
            weight=weight,
        ),
        array(
            [
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
                polyfit(
                    input,
                    other,
                    degree=(0, 1, 2, 3),
                ),
            ]
        ).T,
    )

    assert_array_almost_equal(
        polyfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            1,
        ),
        array([0, 1]),
    )

    assert_array_almost_equal(
        polyfit(
            array([1, 1j, -1, -1j]),
            array([1, 1j, -1, -1j]),
            (0, 1),
        ),
        array([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_array_almost_equal(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=(0, 2, 4),
            ),
        ),
        other,
    )

    assert_array_almost_equal(
        polyfit(
            input,
            other,
            degree=4,
        ),
        polyfit(
            input,
            other,
            degree=(0, 2, 4),
        ),
    )


def test_polyfromroots():
    assert_array_almost_equal(
        polytrim(
            polyfromroots(array([])),
            tol=0.000001,
        ),
        array([1]),
    )

    for i in range(1, 5):
        roots = cos(
            linspace(-math.pi, 0, 2 * i + 1)[1::2],
        )

        assert_array_almost_equal(
            polytrim(
                polyfromroots(roots) * 2 ** (i - 1),
                tol=0.000001,
            ),
            polytrim(
                polycoefficients[i],
                tol=0.000001,
            ),
        )


def test_polygrid2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, array([1.0, 2.0, 3.0]))

    assert_array_almost_equal(
        polygrid2d(
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

    output = polygrid2d(
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
    y = polyval(x, array([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    assert_array_almost_equal(
        polygrid3d(
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

    output = polygrid3d(
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
        polyint(array([0]), 0.5)

    with pytest.raises(ValueError):
        polyint(array([0]), -1)

    with pytest.raises(ValueError):
        polyint(
            array([0]),
            1,
            array([0, 0]),
        )

    with pytest.raises(ValueError):
        polyint(array([0]), lower_bound=[0])

    with pytest.raises(ValueError):
        polyint(array([0]), scale=[0])

    with pytest.raises(TypeError):
        polyint(array([0]), axis=0.5)

    for i in range(2, 5):
        assert_array_almost_equal(
            polytrim(
                polyint(
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
            polytrim(
                polyint(
                    array([0] * i + [1]),
                    order=1,
                    k=[i],
                ),
                tol=0.000001,
            ),
            polytrim(
                array([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_array_almost_equal(
            polyval(
                array([-1]),
                polyint(
                    array([0] * i + [1]),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_array_almost_equal(
            polytrim(
                polyint(
                    array([0] * i + [1]),
                    order=1,
                    k=[i],
                    scale=2,
                ),
                tol=0.000001,
            ),
            polytrim(
                array([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for _ in range(j):
                target = polyint(
                    target,
                    order=1,
                )

            assert_array_almost_equal(
                polytrim(
                    polyint(
                        pol,
                        order=j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                )

            assert_array_almost_equal(
                polytrim(
                    polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            assert_array_almost_equal(
                polytrim(
                    polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = array([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            assert_array_almost_equal(
                polytrim(
                    polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 6))

    assert_array_almost_equal(
        polyint(
            c2d,
            axis=0,
        ),
        vstack([polyint(c) for c in c2d.T]).T,
    )

    assert_array_almost_equal(
        polyint(
            c2d,
            axis=1,
        ),
        vstack([polyint(c) for c in c2d]),
    )

    assert_array_almost_equal(
        polyint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([polyint(c, k=3) for c in c2d]),
    )


def test_polyline():
    assert_array_almost_equal(
        polyline(3, 4),
        array([3, 4]),
    )

    assert_array_almost_equal(
        polyline(3, 0),
        array([3, 0]),
    )


def test_polymul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[i + j].set(target[i + j] + 1)

            assert_array_almost_equal(
                polytrim(
                    polymul(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polymulx():
    assert_array_almost_equal(
        polymulx(
            array([0]),
        ),
        array([0, 0]),
    )

    assert_array_almost_equal(
        polymulx(
            array([1]),
        ),
        array([0, 1]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            polymulx(
                array([0] * i + [1]),
            ),
            array([0] * (i + 1) + [1]),
        )


def test_polyone():
    assert_array_almost_equal(
        polyone,
        array([1]),
    )


def test_polypow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                polytrim(
                    polypow(
                        arange(i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    functools.reduce(
                        polymul,
                        [arange(i + 1)] * j,
                        array([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_polyroots():
    assert_array_almost_equal(
        polyroots(array([1])),
        array([]),
    )

    assert_array_almost_equal(
        polyroots(array([1, 2])),
        array([-0.5]),
    )

    for i in range(2, 5):
        assert_array_almost_equal(
            polytrim(
                polyroots(
                    polyfromroots(
                        linspace(-1, 1, i),
                    ),
                ),
                tol=0.000001,
            ),
            polytrim(
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
                polytrim(
                    polysub(
                        array([0] * i + [1]),
                        array([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polytrim():
    coef = array([2, -1, 1, 0])

    with pytest.raises(ValueError):
        polytrim(coef, -1)

    assert_array_almost_equal(
        polytrim(coef),
        coef[:-1],
    )

    assert_array_almost_equal(
        polytrim(coef, 1),
        coef[:-3],
    )

    assert_array_almost_equal(
        polytrim(coef, 2),
        array([0]),
    )


def test_polyval():
    assert polyval(array([]), array([1])).size == 0

    x = linspace(-1, 1, 50)
    y = [x**i for i in range(5)]

    for i in range(5):
        assert_array_almost_equal(
            polyval(x, array([0] * i + [1])),
            y[i],
        )

    assert_array_almost_equal(
        polyval(x, array([0, -1, 0, 1])),
        x * (x**2 - 1),
    )

    for i in range(3):
        dims = (2,) * i
        x = zeros(dims)

        assert polyval(x, array([1])).shape == dims

        assert polyval(x, array([1, 0])).shape == dims

        assert polyval(x, array([1, 0, 0])).shape == dims


def test_polyval2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    x1, x2, x3 = x

    y1, y2, y3 = polyval(
        x,
        array([1.0, 2.0, 3.0]),
    )

    assert_array_almost_equal(
        polyval2d(
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

    output = polyval2d(
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

    y1, y2, y3 = polyval(
        x,
        array([1.0, 2.0, 3.0]),
    )

    assert_array_almost_equal(
        polyval3d(
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

    output = polyval3d(
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
        polyvalfromroots,
        array([1]),
        array([1]),
        tensor=False,
    )

    assert polyvalfromroots(array([]), array([1])).size == 0

    assert polyvalfromroots(array([]), array([1])).shape == (0,)

    assert polyvalfromroots(array([]), array([[1] * 5])).size == 0

    assert polyvalfromroots(array([]), array([[1] * 5])).shape == (5, 0)

    assert_array_almost_equal(
        polyvalfromroots(
            array([1]),
            array([1]),
        ),
        array([0]),
    )

    assert polyvalfromroots(array([1]), ones((3, 3))).shape == (3, 1)

    input = linspace(-1, 1, 50)
    y = [input**i for i in range(5)]

    for i in range(1, 5):
        target = y[i]

        assert_array_almost_equal(
            polyvalfromroots(
                input,
                array([0] * i),
            ),
            target,
        )

    assert_array_almost_equal(
        polyvalfromroots(
            input,
            array([-1, 0, 1]),
        ),
        input * (input - 1) * (input + 1),
    )

    for i in range(3):
        dims = (2,) * i
        x = zeros(dims)
        assert polyvalfromroots(x, array([1])).shape == dims
        assert polyvalfromroots(x, array([1, 0])).shape == dims
        assert polyvalfromroots(x, array([1, 0, 0])).shape == dims

    ptest = array([15, 2, -16, -2, 1])

    r = polyroots(ptest)

    assert_array_almost_equal(
        polyval(input, ptest),
        polyvalfromroots(input, r),
    )

    x = arange(-3, 2)

    r = jax.random.randint(key, [3, 5], -5, 5)

    target = empty(r.shape[1:])

    for i in range(target.size):
        target = target.at[i].set(polyvalfromroots(x[i], r[:, i]))

    assert_array_almost_equal(
        polyvalfromroots(x, r, tensor=False),
        target,
    )

    x = vstack([x, 2 * x])

    target = empty(r.shape[1:] + x.shape)

    for i in range(r.shape[1]):
        for j in range(x.shape[0]):
            target = target.at[i, j, :].set(polyvalfromroots(x[j], r[:, i]))

    assert_array_almost_equal(
        polyvalfromroots(x, r, tensor=True),
        target,
    )


def test_polyvander():
    output = polyvander(
        arange(3),
        degree=array(3),
    )

    assert output.shape == (3, 4)

    for i in range(4):
        assert_array_almost_equal(
            output[..., i],
            polyval(
                arange(3),
                array([0] * i + [1]),
            ),
        )

    output = polyvander(
        array([[1, 2], [3, 4], [5, 6]]),
        degree=array(3),
    )

    assert output.shape == (3, 2, 4)

    for i in range(4):
        assert_array_almost_equal(
            output[..., i],
            polyval(
                array([[1, 2], [3, 4], [5, 6]]),
                array([0] * i + [1]),
            ),
        )

    with pytest.raises(ValueError):
        polyvander(
            arange(3),
            array([-1]),
        )


def test_polyvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    c = jax.random.uniform(key, (2, 3))

    assert_array_almost_equal(
        dot(
            polyvander2d(
                x1,
                x2,
                degree=array([1, 2]),
            ),
            c.ravel(),
        ),
        polyval2d(
            x1,
            x2,
            c,
        ),
    )

    van = polyvander2d(
        [x1],
        [x2],
        degree=array([1, 2]),
    )

    assert van.shape == (1, 5, 6)


def test_polyvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    c = jax.random.uniform(key, (2, 3, 4))

    assert_array_almost_equal(
        dot(
            polyvander3d(
                x1,
                x2,
                x3,
                degree=array([1, 2, 3]),
            ),
            c.ravel(),
        ),
        polyval3d(
            x1,
            x2,
            x3,
            c,
        ),
    )

    van = polyvander3d(
        [x1],
        [x2],
        [x3],
        degree=array([1, 2, 3]),
    )

    assert van.shape == (1, 5, 24)


def test_polyx():
    assert_array_almost_equal(
        polyx,
        array([0, 1]),
    )


def test_polyzero():
    assert_array_almost_equal(
        polyzero,
        array([0]),
    )
