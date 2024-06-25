import functools
import math

import jax
import numpy
import pytest
from beignet.foo import (
    _c_series_to_z_series,
    _fit,
    _get_domain,
    _map_domain,
    _map_parameters,
    _pow,
    _trim_coefficients,
    _trim_sequence,
    _vander_nd,
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
)
from jax import Array
from torch import (
    arange,
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
    tensor,
    vstack,
    zeros,
    zeros_like,
)
from torch.testing import assert_close

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(0)

chebcoefficients = [
    tensor([1]),
    tensor([0, 1]),
    tensor([-1, 0, 2]),
    tensor([0, -3, 0, 4]),
    tensor([1, 0, -8, 0, 8]),
    tensor([0, 5, 0, -20, 0, 16]),
    tensor([-1, 0, 18, 0, -48, 0, 32]),
    tensor([0, -7, 0, 56, 0, -112, 0, 64]),
    tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]

hermcoefficients = [
    tensor([1]),
    tensor([0, 2]),
    tensor([-2, 0, 4]),
    tensor([0, -12, 0, 8]),
    tensor([12, 0, -48, 0, 16]),
    tensor([0, 120, 0, -160, 0, 32]),
    tensor([-120, 0, 720, 0, -480, 0, 64]),
    tensor([0, -1680, 0, 3360, 0, -1344, 0, 128]),
    tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
    tensor([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
]

hermecoefficients = [
    tensor([1]),
    tensor([0, 1]),
    tensor([-1, 0, 1]),
    tensor([0, -3, 0, 1]),
    tensor([3, 0, -6, 0, 1]),
    tensor([0, 15, 0, -10, 0, 1]),
    tensor([-15, 0, 45, 0, -15, 0, 1]),
    tensor([0, -105, 0, 105, 0, -21, 0, 1]),
    tensor([105, 0, -420, 0, 210, 0, -28, 0, 1]),
    tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
]

lagcoefficients = [
    tensor([1]) / 1,
    tensor([1, -1]) / 1,
    tensor([2, -4, 1]) / 2,
    tensor([6, -18, 9, -1]) / 6,
    tensor([24, -96, 72, -16, 1]) / 24,
    tensor([120, -600, 600, -200, 25, -1]) / 120,
    tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720,
]

legcoefficients = [
    tensor([1]),
    tensor([0, 1]),
    tensor([-1, 0, 3]) / 2,
    tensor([0, -3, 0, 5]) / 2,
    tensor([3, 0, -30, 0, 35]) / 8,
    tensor([0, 15, 0, -70, 0, 63]) / 8,
    tensor([-5, 0, 105, 0, -315, 0, 231]) / 16,
    tensor([0, -35, 0, 315, 0, -693, 0, 429]) / 16,
    tensor([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
    tensor([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
]

polycoefficients = [
    tensor([1]),
    tensor([0, 1]),
    tensor([-1, 0, 2]),
    tensor([0, -3, 0, 4]),
    tensor([1, 0, -8, 0, 8]),
    tensor([0, 5, 0, -20, 0, 16]),
    tensor([-1, 0, 18, 0, -48, 0, 32]),
    tensor([0, -7, 0, 56, 0, -112, 0, 64]),
    tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]


def test__c_series_to_z_series():
    for i in range(5):
        assert_close(
            _c_series_to_z_series(
                tensor(
                    [2] + [1] * i,
                    numpy.double,
                ),
            ),
            tensor(
                [0.5] * i + [2] + [0.5] * i,
                numpy.double,
            ),
        )


def test__map_domain():
    assert_close(
        _map_domain(
            tensor([0, 4]),
            tensor([0, 4]),
            tensor([1, 3]),
        ),
        tensor([1, 3]),
    )

    assert_close(
        _map_domain(
            tensor([0 - 1j, 2 + 1j]),
            tensor([0 - 1j, 2 + 1j]),
            tensor([-2, 2]),
        ),
        tensor([-2, 2]),
    )

    assert_close(
        _map_domain(
            tensor([[0, 4], [0, 4]]),
            tensor([0, 4]),
            tensor([1, 3]),
        ),
        tensor([[1, 3], [1, 3]]),
    )


def test__map_parameters():
    assert_close(
        _map_parameters(
            tensor([0, 4]),
            tensor([1, 3]),
        ),
        tensor([1, 0.5]),
    )

    assert_close(
        _map_parameters(
            tensor([+0 - 1j, +2 + 1j]),
            tensor([-2 + 0j, +2 + 0j]),
        ),
        tensor([-1 + 1j, +1 - 1j]),
    )


def test__pow():
    pytest.raises(
        ValueError,
        _pow,
        (),
        tensor([1, 2, 3]),
        5,
        4,
    )


def test__trim_coefficients():
    pytest.raises(
        ValueError,
        _trim_coefficients,
        tensor([2, -1, 1, 0]),
        -1,
    )

    assert_close(
        _trim_coefficients(
            tensor([2, -1, 1, 0]),
        ),
        tensor([2, -1, 1, 0])[:-1],
    )

    assert_close(
        _trim_coefficients(
            tensor([2, -1, 1, 0]),
            1,
        ),
        tensor([2, -1, 1, 0])[:-3],
    )

    assert_close(
        _trim_coefficients(
            tensor(
                tensor([2, -1, 1, 0]),
            ),
            2,
        ),
        tensor([0]),
    )


def test__trim_sequence():
    for _ in range(5):
        assert_close(
            _trim_sequence(tensor([1] + [0] * 5)),
            tensor([1]),
        )


def test__vandermonde():
    pytest.raises(
        ValueError,
        _vander_nd,
        (),
        (1, 2, 3),
        [90],
    )

    pytest.raises(
        ValueError,
        _vander_nd,
        (),
        (),
        [90.65],
    )

    pytest.raises(
        ValueError,
        _vander_nd,
        (),
        (),
        tensor([]),
    )


def test__z_series_to_c_series():
    for i in range(5):
        assert_close(
            _z_series_to_c_series(
                tensor(
                    [0.5] * i + [2] + [0.5] * i,
                    numpy.float64,
                ),
            ),
            tensor(
                [2] + [1] * i,
                numpy.float64,
            ),
        )


def test_cheb2poly():
    for i in range(10):
        assert_close(
            cheb2poly(
                tensor([0] * i + [1]),
            ),
            chebcoefficients[i],
        )


def test_chebadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_close(
                chebtrim(
                    chebadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
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
        chebcompanion(tensor([]))

    with pytest.raises(ValueError):
        chebcompanion(tensor([1]))

    for i in range(1, 5):
        assert chebcompanion(tensor([0] * i + [1])).shape == (i, i)

    assert chebcompanion(tensor([1, 2]))[0, 0] == -0.5


def test_chebder():
    with pytest.raises(TypeError):
        chebder(tensor([0]), 0.5)

    with pytest.raises(ValueError):
        chebder(tensor([0]), -1)

    for i in range(5):
        assert_close(
            chebtrim(
                chebder(
                    tensor([0] * i + [1]),
                    order=0,
                ),
                tol=0.000001,
            ),
            chebtrim(
                tensor([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                chebtrim(
                    chebder(
                        chebint(
                            tensor(
                                [0] * i + [1],
                            ),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    tensor(
                        [0] * i + [1],
                    ),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                chebtrim(
                    chebder(
                        chebint(
                            tensor([0] * i + [1]),
                            order=j,
                            scl=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    tensor([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_close(
        chebder(c2d, axis=0),
        vstack([chebder(c) for c in c2d.T]).T,
    )

    assert_close(
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
                    tensor([0] * i + [1]),
                    tensor([0] * j + [1]),
                ),
                tensor([0] * i + [1]),
            )

            assert_close(
                chebtrim(
                    chebadd(
                        chebmul(
                            quotient,
                            tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    chebadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebdomain():
    assert_close(
        chebdomain,
        tensor([-1, 1]),
    )


def test__fit():
    with pytest.raises(ValueError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1]),
            degree=-1,
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([[1]]),
            tensor([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([]),
            tensor([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([[[1]]]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([1, 2]),
            tensor([1]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1, 2]),
            degree=0,
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1]),
            degree=0,
            weight=[[1]],
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1]),
            degree=0,
            weight=tensor([1, 1]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1]),
            degree=(-1,),
        )

    with pytest.raises(ValueError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1]),
            degree=(2, -1, 6),
        )

    with pytest.raises(TypeError):
        _fit(
            _vander_nd,
            tensor([1]),
            tensor([1]),
            degree=(),
        )


def test_chebfit():
    def f(x: Array) -> Array:
        return x * (x - 1) * (x - 2)

    def g(x: Array) -> Array:
        return x**4 + x**2 + 1

    x = linspace(0, 2, 50)
    y = f(x)

    coef3 = chebfit(
        x,
        y,
        degree=3,
    )

    assert_close(
        len(coef3),
        4,
    )

    assert_close(
        chebval(
            x,
            coef3,
        ),
        y,
    )

    coef3 = chebfit(
        x,
        y,
        degree=(0, 1, 2, 3),
    )

    assert_close(
        len(coef3),
        4,
    )

    assert_close(
        chebval(x, coef3),
        y,
    )

    coef4 = chebfit(
        x,
        y,
        degree=4,
    )

    assert_close(
        len(coef4),
        5,
    )
    assert_close(
        chebval(x, coef4),
        y,
    )
    coef4 = chebfit(
        x,
        y,
        degree=(0, 1, 2, 3, 4),
    )
    assert_close(
        len(coef4),
        5,
    )
    assert_close(
        chebval(x, coef4),
        y,
    )

    coef4 = chebfit(
        x,
        y,
        degree=(2, 3, 4, 1, 0),
    )
    assert_close(
        len(coef4),
        5,
    )
    assert_close(
        chebval(x, coef4),
        y,
    )

    coef2d = chebfit(
        x,
        tensor([y, y]).T,
        degree=3,
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    coef2d = chebfit(
        x,
        tensor([y, y]).T,
        degree=(0, 1, 2, 3),
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    w = zeros_like(x)
    yw = y.copy()

    w = w.at[1::2].set(1)

    wcoef3 = chebfit(
        x,
        yw,
        degree=3,
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef3 = chebfit(
        x,
        yw,
        degree=(0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef2d = chebfit(
        x,
        tensor([yw, yw]).T,
        degree=3,
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    wcoef2d = chebfit(
        x,
        tensor([yw, yw]).T,
        degree=(0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor(
            [
                coef3,
                coef3,
            ],
        ).T,
    )

    assert_close(
        chebfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=1,
        ),
        tensor([0, 1]),
    )

    assert_close(
        chebfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=(0, 1),
        ),
        tensor([0, 1]),
    )

    input = linspace(-1, 1, 50)

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                g(input),
                degree=4,
            ),
        ),
        g(input),
    )

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                g(input),
                (0, 2, 4),
            ),
        ),
        g(input),
    )

    assert_close(
        chebfit(
            input,
            g(input),
            degree=4,
        ),
        chebfit(
            input,
            g(input),
            (0, 2, 4),
        ),
    )


def test_chebfromroots():
    assert_close(
        chebtrim(
            chebfromroots(
                tensor([]),
            ),
            tol=0.000001,
        ),
        tensor([1]),
    )
    for i in range(1, 5):
        assert_close(
            chebtrim(
                chebfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2]))
                * 2 ** (i - 1),
                tol=0.000001,
            ),
            chebtrim(
                tensor([0] * i + [1]),
                tol=0.000001,
            ),
        )


def test_chebgauss():
    x, w = chebgauss(100)

    v = chebvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_close(vv, eye(100))
    assert_close(w.sum(), math.pi)


def test_chebgrid2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
        chebgrid2d(
            x1,
            x2,
            einsum("i,j->ij", tensor([2.5, 2.0, 1.5]), tensor([2.5, 2.0, 1.5])),
        ),
        einsum("i,j->ij", y1, y2),
    )

    z = ones([2, 3])
    res = chebgrid2d(
        z,
        z,
        einsum("i,j->ij", tensor([2.5, 2.0, 1.5]), tensor([2.5, 2.0, 1.5])),
    )
    assert res.shape == (2, 3) * 2


def test_chebgrid3d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
        chebgrid3d(
            x1,
            x2,
            x3,
            einsum(
                "i,j,k->ijk",
                tensor([2.5, 2.0, 1.5]),
                tensor([2.5, 2.0, 1.5]),
                tensor([2.5, 2.0, 1.5]),
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
    res = chebgrid3d(
        z,
        z,
        z,
        einsum(
            "i,j,k->ijk",
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
        ),
    )
    assert res.shape == (2, 3) * 3


def test_chebint():
    pytest.raises(TypeError, chebint, tensor([0]), 0.5)
    pytest.raises(ValueError, chebint, tensor([0]), -1)
    pytest.raises(ValueError, chebint, tensor([0]), 1, [0, 0])
    pytest.raises(ValueError, chebint, tensor([0]), lbnd=[0])
    pytest.raises(ValueError, chebint, tensor([0]), scl=[0])
    pytest.raises(TypeError, chebint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_close(
            chebtrim(chebint(tensor([0]), order=i, k=k), tol=0.000001),
            [0, 1],
        )

    for i in range(5):
        assert_close(
            chebtrim(
                cheb2poly(chebint(poly2cheb(tensor([0] * i + [1])), order=1, k=[i])),
                tol=0.000001,
            ),
            chebtrim(tensor([i] + [0] * i + [1 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        assert_close(
            chebval(
                tensor([-1]),
                chebint(
                    poly2cheb(tensor([0] * i + [1])),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_close(
            chebtrim(
                cheb2poly(
                    chebint(
                        poly2cheb(tensor([0] * i + [1])),
                        order=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tol=0.000001,
            ),
            chebtrim(tensor([i] + [0] * i + [2 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = chebint(target, order=1)
            assert_close(
                chebtrim(chebint(pol, order=j), tol=0.000001),
                chebtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]

            for k in range(j):
                target = chebint(target, order=1, k=[k])

            assert_close(
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
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = chebint(target, order=1, k=[k], lbnd=-1)

            assert_close(
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
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = chebint(target, order=1, k=[k], scl=2)
            assert_close(
                chebtrim(
                    chebint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                chebtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([chebint(c) for c in c2d.T]).T
    assert_close(
        chebint(c2d, axis=0),
        target,
    )

    target = vstack([chebint(c) for c in c2d])
    assert_close(
        chebint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = vstack([chebint(c, k=3) for c in c2d])
    assert_close(
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
            assert_close(
                chebval(x, c),
                powx(x, p),
            )


def test_chebline():
    assert_close(chebline(3, 4), tensor([3, 4]))


def test_chebmul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[abs(i + j)].set(target[abs(i + j)] + 0.5)
            target = target.at[abs(i - j)].set(target[abs(i - j)] + 0.5)

            assert_close(
                chebtrim(
                    chebmul(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebmulx():
    assert_close(chebtrim(chebmulx([0]), tol=0.000001), [0])

    assert_close(chebtrim(chebmulx([1]), tol=0.000001), [0, 1])

    for i in range(1, 5):
        assert_close(
            chebtrim(chebmulx(tensor([0] * i + [1])), tol=0.000001),
            [0] * (i - 1) + [0.5, 0, 0.5],
        )


def test_chebone():
    assert_close(
        chebone,
        tensor([1]),
    )


def test_chebpow():
    for i in range(5):
        for j in range(5):
            assert_close(
                chebtrim(chebpow(arange(i + 1), j), tol=0.000001),
                chebtrim(
                    functools.reduce(
                        chebmul,
                        [(arange(i + 1))] * j,
                        tensor([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebpts1():
    with pytest.raises(ValueError):
        chebpts1(1.5)

    with pytest.raises(ValueError):
        chebpts1(0)

    assert_close(chebpts1(1), [0])

    assert_close(chebpts1(2), [-0.70710678118654746, 0.70710678118654746])

    assert_close(chebpts1(3), [-0.86602540378443871, 0, 0.86602540378443871])

    assert_close(
        chebpts1(4),
        [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325],
    )


def test_chebpts2():
    with pytest.raises(ValueError):
        chebpts2(1.5)

    with pytest.raises(ValueError):
        chebpts2(1)

    assert_close(chebpts2(2), [-1, 1])

    assert_close(chebpts2(3), tensor([-1, 0, 1]))

    assert_close(chebpts2(4), [-1, -0.5, 0.5, 1])

    assert_close(chebpts2(5), [-1.0, -0.707106781187, 0, 0.707106781187, 1.0])


def test_chebroots():
    assert_close(
        chebroots([1]),
        tensor([]),
    )

    assert_close(
        chebroots(tensor([1, 2])),
        tensor([-0.5]),
    )

    for i in range(2, 5):
        assert_close(
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

            assert_close(
                chebtrim(
                    chebsub(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebtrim():
    pytest.raises(ValueError, chebtrim, tensor([2, -1, 1, 0]), -1)

    assert_close(chebtrim(tensor([2, -1, 1, 0])), tensor([2, -1, 1, 0])[:-1])

    assert_close(chebtrim(tensor([2, -1, 1, 0]), 1), tensor([2, -1, 1, 0])[:-3])

    assert_close(
        chebtrim(tensor([2, -1, 1, 0]), 2),
        tensor([0]),
    )


def test_chebval():
    assert_close(chebval(tensor([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [polyval(x, c) for c in chebcoefficients]
    for i in range(10):
        target = y[i]
        res = chebval(x, tensor([0] * i + [1]))
        assert_close(
            res,
            target,
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_close(chebval(x, [1]).shape, dims)
        assert_close(chebval(x, tensor([1, 0])).shape, dims)
        assert_close(chebval(x, tensor([1, 0, 0])).shape, dims)


def test_chebval2d():
    c1d = tensor([2.5, 2.0, 1.5])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

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
    assert_close(
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
    c1d = tensor([2.5, 2.0, 1.5])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, chebval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    res = chebval3d(x1, x2, x3, c3d)
    assert_close(
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
        coef = tensor([0] * i + [1])
        assert_close(v[..., i], chebval(x, coef))

    x = tensor([[1, 2], [3, 4], [5, 6]])
    v = chebvander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = tensor([0] * i + [1])
        assert_close(v[..., i], chebval(x, coef))


def test_chebvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    van = chebvander2d(x1, x2, (1, 2))

    assert_close(
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
    assert_close(
        dot(chebvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        chebval3d(x1, x2, x3, c),
    )

    van = chebvander3d([x1], [x2], [x3], (1, 2, 3))
    assert van.shape == (1, 5, 24)


def test_chebweight():
    x = linspace(-1, 1, 11)[1:-1]
    assert_close(
        chebweight(x),
        1.0 / (sqrt(1 + x) * sqrt(1 - x)),
    )


def test_chebx():
    assert_close(
        chebx,
        tensor([0, 1]),
    )


def test_chebzero():
    assert_close(chebzero, tensor([0]))


def test__get_domain():
    assert_close(
        _get_domain(tensor([1, 10, 3, -1])),
        tensor([-1, 10]),
    )

    assert_close(
        _get_domain(tensor([1 + 1j, 1 - 1j, 0, 2])),
        tensor([-1j, 2 + 1j]),
    )


def test_herm2poly():
    for i in range(10):
        assert_close(
            herm2poly(tensor([0] * i + [1])),
            hermcoefficients[i],
        )


def test_hermadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_close(
                hermtrim(
                    hermadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
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
        hermcompanion(tensor([]))
    with pytest.raises(ValueError):
        hermcompanion([1])

    for i in range(1, 5):
        assert hermcompanion(tensor([0] * i + [1])).shape == (i, i)

    assert hermcompanion(tensor([1, 2]))[0, 0] == -0.25


def test_hermder():
    with pytest.raises(TypeError):
        hermder(tensor([0]), 0.5)

    with pytest.raises(ValueError):
        hermder(tensor([0]), -1)

    for i in range(5):
        target = tensor([0] * i + [1])
        res = hermder(target, order=0)
        assert_close(
            hermtrim(res, tol=0.000001),
            hermtrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            target = tensor([0] * i + [1])
            res = hermder(hermint(target, order=j), order=j)
            assert_close(
                hermtrim(res, tol=0.000001),
                hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            target = tensor([0] * i + [1])
            res = hermder(hermint(target, order=j, scl=2), order=j, scl=0.5)
            assert_close(
                hermtrim(res, tol=0.000001),
                hermtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([hermder(c) for c in c2d.T]).T
    res = hermder(c2d, axis=0)
    assert_close(
        res,
        target,
    )

    target = vstack([hermder(c) for c in c2d])
    res = hermder(
        c2d,
        axis=1,
    )
    assert_close(
        res,
        target,
    )


def test_hermdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = hermdiv(
                hermadd(
                    tensor([0.0] * i + [1.0]),
                    tensor([0.0] * j + [1.0]),
                ),
                tensor([0.0] * i + [1.0]),
            )

            assert_close(
                hermtrim(
                    hermadd(
                        hermmul(
                            quotient,
                            tensor([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    hermadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermdomain():
    assert_close(hermdomain, tensor([-1, 1]))


def test_herme2poly():
    for i in range(10):
        assert_close(
            herme2poly(tensor([0] * i + [1])),
            hermecoefficients[i],
        )


def test_hermeadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_close(
                hermetrim(
                    hermeadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
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
        hermecompanion(tensor([]))

    with pytest.raises(ValueError):
        hermecompanion([1])

    for i in range(1, 5):
        coef = tensor([0] * i + [1])
        assert hermecompanion(coef).shape == (i, i)

    assert hermecompanion(tensor([1, 2]))[0, 0] == -0.5


def test_hermeder():
    pytest.raises(TypeError, hermeder, tensor([0]), 0.5)
    pytest.raises(ValueError, hermeder, tensor([0]), -1)

    for i in range(5):
        assert_close(
            hermetrim(hermeder(tensor([0] * i + [1]), order=0), tol=0.000001),
            hermetrim(tensor([0] * i + [1]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                hermetrim(
                    hermeder(
                        hermeint(tensor([0] * i + [1]), order=j),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermetrim(tensor([0] * i + [1]), tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                hermetrim(
                    hermeder(
                        hermeint(tensor([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                hermetrim(tensor([0] * i + [1]), tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_close(
        hermeder(c2d, axis=0),
        vstack([hermeder(c) for c in c2d.T]).T,
    )

    assert_close(
        hermeder(
            c2d,
            axis=1,
        ),
        vstack([hermeder(c) for c in c2d]),
    )


def test_hermediv():
    for i in range(5):
        for j in range(5):
            ci = tensor([0] * i + [1])
            cj = tensor([0] * j + [1])
            quotient, remainder = hermediv(hermeadd(ci, cj), ci)
            assert_close(
                hermetrim(
                    hermeadd(hermemul(quotient, ci), remainder),
                    tol=0.000001,
                ),
                hermetrim(hermeadd(ci, cj), tol=0.000001),
            )


def test_hermedomain():
    assert_close(hermedomain, [-1, 1])


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    x = linspace(0, 2)

    y = f(x)

    coef3 = hermefit(
        x,
        y,
        3,
    )

    assert_close(
        len(coef3),
        4,
    )

    assert_close(
        hermeval(x, coef3),
        y,
    )

    coef3 = hermefit(
        x,
        y,
        (0, 1, 2, 3),
    )

    assert_close(
        len(coef3),
        4,
    )

    assert_close(
        hermeval(x, coef3),
        y,
    )

    coef4 = hermefit(
        x,
        y,
        4,
    )

    assert_close(
        len(coef4),
        5,
    )

    assert_close(
        hermeval(x, coef4),
        y,
    )

    coef4 = hermefit(
        x,
        y,
        (0, 1, 2, 3, 4),
    )

    assert_close(
        len(coef4),
        5,
    )

    assert_close(
        hermeval(x, coef4),
        y,
    )

    coef4 = hermefit(
        x,
        y,
        (2, 3, 4, 1, 0),
    )

    assert_close(
        len(coef4),
        5,
    )

    assert_close(
        hermeval(x, coef4),
        y,
    )

    coef2d = hermefit(
        x,
        tensor([y, y]).T,
        3,
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    coef2d = hermefit(
        x,
        tensor([y, y]).T,
        (0, 1, 2, 3),
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    w = zeros_like(x)

    yw = y.copy()

    w = w.at[1::2].set(1)
    y = y.at[0::2].set(0)

    wcoef3 = hermefit(
        x,
        yw,
        3,
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef3 = hermefit(
        x,
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef2d = hermefit(
        x,
        tensor([yw, yw]).T,
        3,
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    wcoef2d = hermefit(
        x,
        tensor([yw, yw]).T,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    x = tensor([1, 1j, -1, -1j])

    assert_close(
        hermefit(x, x, 1),
        tensor([0, 1]),
    )

    assert_close(
        hermefit(x, x, (0, 1)),
        tensor([0, 1]),
    )

    x = linspace(-1, 1)

    y = g(x)

    coef1 = hermefit(x, y, 4)

    assert_close(
        hermeval(x, coef1),
        y,
    )

    assert_close(
        hermeval(x, hermefit(x, y, (0, 2, 4))),
        y,
    )

    assert_close(
        coef1,
        hermefit(x, y, (0, 2, 4)),
    )


def test_hermefromroots():
    res = hermefromroots(tensor([]))
    assert_close(
        hermetrim(res, tol=0.000001),
        tensor([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = hermefromroots(roots)
        assert len(pol) == i + 1
        assert_close(herme2poly(pol)[-1], 1)
        assert_close(
            hermeval(roots, pol),
            0,
        )


def test_hermegauss():
    x, w = hermegauss(100)

    v = hermevander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_close(vv, eye(100))

    target = sqrt(2 * math.pi)
    assert_close(w.sum(), target)


def test_hermegrid2d():
    c1d = tensor([4.0, 2.0, 3.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j->ij", y1, y2)
    res = hermegrid2d(
        x1,
        x2,
        c2d,
    )
    assert_close(
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
    c1d = tensor([4.0, 2.0, 3.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    assert_close(
        hermegrid3d(x1, x2, x3, c3d),
        einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = ones([2, 3])
    res = hermegrid3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3


def test_hermeint():
    pytest.raises(TypeError, hermeint, tensor([0]), 0.5)
    pytest.raises(ValueError, hermeint, tensor([0]), -1)
    pytest.raises(ValueError, hermeint, tensor([0]), 1, [0, 0])
    pytest.raises(ValueError, hermeint, tensor([0]), lbnd=[0])
    pytest.raises(ValueError, hermeint, tensor([0]), scl=[0])
    pytest.raises(TypeError, hermeint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = hermeint(tensor([0]), order=i, k=k)
        assert_close(
            hermetrim(res, tol=0.000001),
            tensor([0, 1]),
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        target = [i] + [0] * i + [1 / scl]
        hermepol = poly2herme(pol)
        res = herme2poly(hermeint(hermepol, order=1, k=[i]))
        assert_close(
            hermetrim(res, tol=0.000001),
            hermetrim(target, tol=0.000001),
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        hermepol = poly2herme(pol)
        assert_close(
            hermeval(
                tensor([-1]),
                hermeint(hermepol, order=1, k=[i], lbnd=-1),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        target = [i] + [0] * i + [2 / scl]
        hermepol = poly2herme(pol)
        res = herme2poly(hermeint(hermepol, order=1, k=[i], scl=2))
        assert_close(
            hermetrim(res, tol=0.000001),
            hermetrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = hermeint(target, order=1)
            res = hermeint(pol, order=j)
            assert_close(
                hermetrim(res, tol=0.000001),
                hermetrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermeint(target, order=1, k=[k])

            assert_close(
                hermetrim(hermeint(pol, order=j, k=list(range(j))), tol=0.000001),
                hermetrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermeint(target, order=1, k=[k], lbnd=-1)
            assert_close(
                hermetrim(
                    hermeint(pol, order=j, k=list(range(j)), lbnd=-1), tol=0.000001
                ),
                hermetrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermeint(target, order=1, k=[k], scl=2)
            assert_close(
                hermetrim(
                    hermeint(pol, order=j, k=list(range(j)), scl=2), tol=0.000001
                ),
                hermetrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_close(
        hermeint(c2d, axis=0),
        vstack([hermeint(c) for c in c2d.T]).T,
    )

    target = vstack([hermeint(c) for c in c2d])
    res = hermeint(
        c2d,
        axis=1,
    )
    assert_close(
        res,
        target,
    )

    target = vstack([hermeint(c, k=3) for c in c2d])
    res = hermeint(
        c2d,
        k=3,
        axis=1,
    )
    assert_close(
        res,
        target,
    )


def test_hermeline():
    assert_close(hermeline(3, 4), tensor([3, 4]))


def test_hermemul():
    x = linspace(-3, 3, 100)
    for i in range(5):
        pol1 = tensor([0] * i + [1])
        val1 = hermeval(x, pol1)
        for j in range(5):
            pol2 = tensor([0] * j + [1])
            val2 = hermeval(x, pol2)
            pol3 = hermemul(pol1, pol2)
            val3 = hermeval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_close(
                val3,
                val1 * val2,
            )


def test_hermemulx():
    assert_close(hermetrim(hermemulx([0]), tol=0.000001), [0])
    assert_close(hermetrim(hermemulx([1]), tol=0.000001), [0, 1])
    for i in range(1, 5):
        assert_close(
            hermetrim(hermemulx(tensor([0] * i + [1])), tol=0.000001),
            [0] * (i - 1) + [i, 0, 1],
        )


def test_hermeone():
    assert_close(
        hermeone,
        tensor([1]),
    )


def test_hermepow():
    for i in range(5):
        for j in range(5):
            assert_close(
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
                        tensor([arange(i + 1)] * j),
                        tensor([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermeroots():
    assert_close(hermeroots([1]), tensor([]))

    assert_close(hermeroots([1, 1]), [-1])

    for i in range(2, 5):
        assert_close(
            hermetrim(
                hermeroots(
                    hermefromroots(
                        linspace(-1, 1, i),
                    )
                ),
                tol=0.000001,
            ),
            hermetrim(linspace(-1, 1, i), tol=0.000001),
        )


def test_hermesub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_close(
                hermetrim(
                    hermesub(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermetrim():
    coef = tensor([2, -1, 1, 0])

    pytest.raises(ValueError, hermetrim, coef, -1)

    assert_close(hermetrim(coef), coef[:-1])
    assert_close(hermetrim(coef, 1), coef[:-3])
    assert_close(hermetrim(coef, 2), [0])


def test_hermeval():
    assert_close(hermeval(tensor([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [polyval(x, c) for c in hermecoefficients]
    for i in range(10):
        assert_close(hermeval(x, tensor([0] * i + [1])), y[i], decimal=4)

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_close(hermeval(x, [1]).shape, dims)
        assert_close(hermeval(x, tensor([1, 0])).shape, dims)
        assert_close(hermeval(x, tensor([1, 0, 0])).shape, dims)


def test_hermeval2d():
    c1d = tensor([4.0, 2.0, 3.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    pytest.raises(
        ValueError,
        hermeval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_close(
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
    c1d = tensor([4.0, 2.0, 3.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, hermeval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    res = hermeval3d(x1, x2, x3, c3d)
    assert_close(res, target, decimal=4)

    z = ones([2, 3])
    res = hermeval3d(z, z, z, c3d)
    assert res.shape == (2, 3)


def test_hermevander():
    x = arange(3)
    v = hermevander(x, 3)
    assert v.shape == (3, 4)
    for i in range(4):
        coef = tensor([0] * i + [1])
        assert_close(v[..., i], hermeval(x, coef))

    x = tensor([[1, 2], [3, 4], [5, 6]])
    v = hermevander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = tensor([0] * i + [1])
        assert_close(v[..., i], hermeval(x, coef))


def test_hermevander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_close(
        dot(hermevander2d(x1, x2, (1, 2)), c.ravel()),
        hermeval2d(x1, x2, c),
    )

    van = hermevander2d([x1], [x2], (1, 2))
    assert van.shape == (1, 5, 6)


def test_hermevander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    van = hermevander3d(x1, x2, x3, (1, 2, 3))
    assert_close(dot(van, c.ravel()), hermeval3d(x1, x2, x3, c))

    van = hermevander3d([x1], [x2], [x3], (1, 2, 3))
    assert van.shape == (1, 5, 24)


def test_hermeweight():
    x = linspace(-5, 5, 11)
    target = exp(-0.5 * x**2)
    res = hermeweight(x)
    assert_close(
        res,
        target,
    )


def test_hermex():
    assert_close(
        hermex,
        tensor([0, 1]),
    )


def test_hermezero():
    assert_close(
        hermezero,
        tensor([0]),
    )


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    x = linspace(0, 2)

    y = f(x)

    coef3 = hermfit(
        x,
        y,
        3,
    )

    assert len(coef3) == 4

    assert_close(
        hermval(x, coef3),
        y,
    )

    coef3 = hermfit(
        x,
        y,
        (0, 1, 2, 3),
    )

    assert len(coef3) == 4

    assert_close(
        hermval(x, coef3),
        y,
    )

    coef4 = hermfit(
        x,
        y,
        4,
    )

    assert len(coef4) == 5

    assert_close(
        hermval(x, coef4),
        y,
    )

    coef4 = hermfit(
        x,
        y,
        (0, 1, 2, 3, 4),
    )

    assert len(coef4) == 5

    assert_close(
        hermval(x, coef4),
        y,
    )

    coef4 = hermfit(
        x,
        y,
        (2, 3, 4, 1, 0),
    )

    assert len(coef4) == 5

    assert_close(
        hermval(x, coef4),
        y,
    )

    coef2d = hermfit(
        x,
        tensor([y, y]).T,
        3,
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    coef2d = hermfit(
        x,
        tensor([y, y]).T,
        (0, 1, 2, 3),
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    w = zeros_like(x)

    yw = y.copy()

    w = w.at[1::2].set(1)
    y = y.at[0::2].set(0)

    wcoef3 = hermfit(
        x,
        yw,
        3,
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef3 = hermfit(
        x,
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef2d = hermfit(
        x,
        tensor([yw, yw]).T,
        3,
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    wcoef2d = hermfit(
        x,
        tensor([yw, yw]).T,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    x = tensor([1, 1j, -1, -1j])

    assert_close(
        hermfit(x, x, 1),
        tensor([0, 0.5]),
    )

    assert_close(
        hermfit(x, x, (0, 1)),
        tensor([0, 0.5]),
    )

    x = linspace(-1, 1)

    y = g(x)

    coef1 = hermfit(
        x,
        y,
        4,
    )

    assert_close(
        hermval(x, coef1),
        y,
    )

    coef2 = hermfit(
        x,
        y,
        (0, 2, 4),
    )

    assert_close(
        hermval(x, coef2),
        y,
    )

    assert_close(
        coef1,
        coef2,
    )


def test_hermfromroots():
    res = hermfromroots(tensor([]))
    assert_close(
        hermtrim(res, tol=0.000001),
        tensor([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = hermfromroots(roots)
        res = hermval(roots, pol)
        target = 0
        assert len(pol) == i + 1
        assert_close(herm2poly(pol)[-1], 1)
        assert_close(
            res,
            target,
        )


def test_hermgauss():
    x, w = hermgauss(100)

    v = hermvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_close(
        vv,
        eye(100),
    )

    target = sqrt(math.pi)
    assert_close(w.sum(), target)


def test_hermgrid2d():
    c1d = tensor([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    target = einsum("i,j->ij", y1, y2)
    assert_close(
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
    c1d = tensor([2.5, 1.0, 0.75])
    c3d = einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
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
    pytest.raises(TypeError, hermint, tensor([0]), 0.5)
    pytest.raises(ValueError, hermint, tensor([0]), -1)
    pytest.raises(
        ValueError,
        hermint,
        tensor([0]),
        1,
        tensor([0, 0]),
    )
    pytest.raises(ValueError, hermint, tensor([0]), lbnd=[0])
    pytest.raises(ValueError, hermint, tensor([0]), scl=[0])
    pytest.raises(TypeError, hermint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_close(
            hermtrim(hermint(tensor([0]), order=i, k=k), tol=0.000001),
            [0, 0.5],
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        hermpol = poly2herm(pol)
        assert_close(
            hermtrim(herm2poly(hermint(hermpol, order=1, k=[i])), tol=0.000001),
            hermtrim([i] + [0] * i + [1 / scl], tol=0.000001),
        )

    for i in range(5):
        pol = tensor([0] * i + [1])
        hermpol = poly2herm(pol)
        assert_close(hermval(tensor(-1), hermint(hermpol, order=1, k=[i], lbnd=-1)), i)

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        hermpol = poly2herm(pol)
        assert_close(
            hermtrim(herm2poly(hermint(hermpol, order=1, k=[i], scl=2)), tol=0.000001),
            hermtrim([i] + [0] * i + [2 / scl], tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = hermint(target, order=1)
            assert_close(
                hermtrim(hermint(pol, order=j), tol=0.000001),
                hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermint(target, order=1, k=[k])
            assert_close(
                hermtrim(hermint(pol, order=j, k=list(range(j))), tol=0.000001),
                hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermint(target, order=1, k=[k], lbnd=-1)
            assert_close(
                hermtrim(
                    hermint(pol, order=j, k=list(range(j)), lbnd=-1),
                    tol=0.000001,
                ),
                hermtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = hermint(target, order=1, k=[k], scl=2)
            assert_close(
                hermtrim(
                    hermint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                hermtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([hermint(c) for c in c2d.T]).T
    assert_close(hermint(c2d, axis=0), target)

    target = vstack([hermint(c) for c in c2d])
    assert_close(
        hermint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = vstack([hermint(c, k=3) for c in c2d])
    assert_close(
        hermint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )


def test_hermline():
    assert_close(hermline(3, 4), [3, 2])


def test_hermmul():
    x = linspace(-3, 3, 100)

    for i in range(5):
        pol1 = tensor([0.0] * i + [1.0])
        val1 = hermval(x, pol1)
        for j in range(5):
            pol2 = tensor([0.0] * j + [1.0])
            val2 = hermval(x, pol2)
            pol3 = hermmul(pol1, pol2)
            val3 = hermval(x, pol3)

            assert len(hermtrim(pol3, tol=0.000001)) == i + j + 1

            assert_close(
                val3,
                val1 * val2,
            )


def test_hermmulx():
    assert_close(hermtrim(hermmulx([0.0]), tol=0.000001), [0.0])
    assert_close(hermmulx([1.0]), [0.0, 0.5])
    for i in range(1, 5):
        assert_close(
            hermmulx(tensor([0.0] * i + [1.0])),
            [0.0] * (i - 1) + [i, 0.0, 0.5],
        )


def test_hermone():
    assert_close(hermone, tensor([1]))


def test_hermpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1).astype(float)
            assert_close(
                hermtrim(hermpow(c, j), tol=0.000001),
                hermtrim(
                    functools.reduce(hermmul, [c] * j, tensor([1])),
                    tol=0.000001,
                ),
            )


def test_hermroots():
    assert_close(hermroots(tensor([1])), tensor([]))
    assert_close(hermroots(tensor([1, 1])), tensor([-0.5]))
    for i in range(2, 5):
        target = linspace(-1, 1, i)
        assert_close(
            hermtrim(
                hermroots(hermfromroots(target)),
                tol=0.000001,
            ),
            hermtrim(target, tol=0.000001),
        )


def test_hermsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_close(
                hermtrim(
                    hermsub(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermtrim():
    coef = tensor([2, -1, 1, 0])

    pytest.raises(ValueError, hermtrim, coef, -1)

    assert_close(hermtrim(coef), coef[:-1])
    assert_close(hermtrim(coef, 1), coef[:-3])
    assert_close(
        hermtrim(coef, 2),
        tensor([0]),
    )


def test_hermval():
    assert hermval(tensor([]), [1]).size == 0

    x = linspace(-1, 1)
    y = [polyval(x, c) for c in hermcoefficients]

    for index in range(10):
        assert_close(
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
        assert hermval(x, tensor([1, 0])).shape == dims
        assert hermval(x, tensor([1, 0, 0])).shape == dims


def test_hermval2d():
    c1d = tensor([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        hermval2d,
        x1,
        x2[:2],
        c2d,
    )

    target = y1 * y2
    res = hermval2d(
        x1,
        x2,
        c2d,
    )
    assert_close(
        res,
        target,
    )

    z = ones([2, 3])
    res = hermval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_hermval3d():
    c1d = tensor([2.5, 1.0, 0.75])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, hermval3d, x1, x2, x3[:2], c3d)

    target = y1 * y2 * y3
    assert_close(
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
        coef = tensor([0] * i + [1])
        assert_close(v[..., i], hermval(x, coef))

    x = tensor([[1, 2], [3, 4], [5, 6]])
    v = hermvander(x, 3)
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coef = tensor([0] * i + [1])
        assert_close(v[..., i], hermval(x, coef))


def test_hermvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_close(
        dot(hermvander2d(x1, x2, (1, 2)), c.ravel()),
        hermval2d(x1, x2, c),
    )

    assert hermvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_hermvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_close(
        dot(hermvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        hermval3d(x1, x2, x3, c),
    )

    assert hermvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_hermweight():
    assert_close(
        hermweight(linspace(-5, 5, 11)),
        exp(-(linspace(-5, 5, 11) ** 2)),
    )


def test_hermx():
    assert_close(hermx, tensor([0, 0.5]))


def test_hermzero():
    assert_close(hermzero, tensor([0]))


def test_lag2poly():
    for i in range(7):
        assert_close(lag2poly(tensor([0] * i + [1])), lagcoefficients[i])


def test_lagadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_close(
                lagtrim(
                    lagadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
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
        lagcompanion(tensor([]))
    with pytest.raises(ValueError):
        lagcompanion([1])

    for i in range(1, 5):
        coef = tensor([0] * i + [1])
        assert lagcompanion(coef).shape == (i, i)

    assert lagcompanion(tensor([1, 2]))[0, 0] == 1.5


def test_lagder():
    pytest.raises(TypeError, lagder, tensor([0]), 0.5)
    pytest.raises(ValueError, lagder, tensor([0]), -1)

    for i in range(5):
        assert_close(
            lagtrim(lagder(tensor([0] * i + [1]), order=0), tol=0.000001),
            lagtrim(tensor([0] * i + [1]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                lagtrim(
                    lagder(lagint(tensor([0] * i + [1]), order=j), order=j),
                    tol=0.000001,
                ),
                lagtrim(tensor([0] * i + [1]), tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                lagtrim(
                    lagder(
                        lagint(tensor([0] * i + [1]), order=j, scl=2),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                lagtrim(tensor([0] * i + [1]), tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_close(
        lagder(c2d, axis=0),
        vstack([lagder(c) for c in c2d.T]).T,
    )

    assert_close(
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
                    tensor([0] * i + [1]),
                    tensor([0] * j + [1]),
                ),
                tensor([0] * i + [1]),
            )

            assert_close(
                lagtrim(
                    lagadd(
                        lagmul(
                            quotient,
                            tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    lagadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagdomain():
    assert_close(
        lagdomain,
        tensor([0, 1]),
    )


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    x = linspace(0, 2)
    y = f(x)

    coef3 = lagfit(
        x,
        y,
        3,
    )

    assert_close(
        len(coef3),
        4,
    )

    assert_close(
        lagval(x, coef3),
        y,
    )

    coef3 = lagfit(
        x,
        y,
        (0, 1, 2, 3),
    )

    assert_close(
        len(coef3),
        4,
    )

    assert_close(
        lagval(x, coef3),
        y,
    )

    coef4 = lagfit(
        x,
        y,
        4,
    )

    assert_close(
        len(coef4),
        5,
    )

    assert_close(
        lagval(x, coef4),
        y,
    )

    coef4 = lagfit(
        x,
        y,
        (0, 1, 2, 3, 4),
    )

    assert_close(
        len(coef4),
        5,
    )

    assert_close(
        lagval(x, coef4),
        y,
    )

    coef2d = lagfit(
        x,
        tensor([y, y]).T,
        3,
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    coef2d = lagfit(
        x,
        tensor([y, y]).T,
        (0, 1, 2, 3),
    )

    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    w = zeros_like(x)

    yw = y.copy()

    w = w.at[1::2].set(1)
    y = y.at[0::2].set(0)

    wcoef3 = lagfit(
        x,
        yw,
        3,
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef3 = lagfit(
        x,
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef2d = lagfit(
        x,
        tensor([yw, yw]).T,
        3,
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    wcoef2d = lagfit(
        x,
        tensor([yw, yw]).T,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef2d,
        tensor([coef3, coef3]).T,
    )

    x = tensor([1, 1j, -1, -1j])

    assert_close(
        lagfit(x, x, 1),
        tensor([1, -1]),
    )

    assert_close(
        lagfit(x, x, (0, 1)),
        tensor([1, -1]),
    )


def test_lagfromroots():
    res = lagfromroots(tensor([]))
    assert_close(
        lagtrim(res, tol=0.000001),
        tensor([1]),
    )
    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        pol = lagfromroots(roots)
        res = lagval(roots, pol)
        target = 0
        assert len(pol) == i + 1
        assert_close(lag2poly(pol)[-1], 1)
        assert_close(res, target, decimal=4)


def test_laggauss():
    x, w = laggauss(100)

    v = lagvander(x, 99)
    vv = dot(v.T * w, v)
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_close(
        vv,
        eye(100),
    )

    target = 1.0
    assert_close(w.sum(), target)


def test_laggrid2d():
    c1d = tensor([9.0, -14.0, 6.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
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
    c1d = tensor([9.0, -14.0, 6.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j,k->ijk", y1, y2, y3)
    assert_close(laggrid3d(x1, x2, x3, c3d), target, decimal=3)

    z = ones([2, 3])
    assert laggrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_lagint():
    pytest.raises(TypeError, lagint, tensor([0]), 0.5)
    pytest.raises(ValueError, lagint, tensor([0]), -1)
    pytest.raises(
        ValueError,
        lagint,
        tensor([0]),
        1,
        tensor([0, 0]),
    )
    pytest.raises(ValueError, lagint, tensor([0]), lbnd=[0])
    pytest.raises(ValueError, lagint, tensor([0]), scl=[0])
    pytest.raises(TypeError, lagint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_close(
            lagtrim(lagint(tensor([0]), order=i, k=k), tol=0.000001),
            [1, -1],
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        target = [i] + [0] * i + [1 / scl]
        res = lag2poly(lagint(poly2lag(pol), order=1, k=[i]))
        assert_close(
            lagtrim(res, tol=0.000001),
            lagtrim(target, tol=0.000001),
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        lagpol = poly2lag(pol)
        assert_close(
            lagval(
                tensor([-1]),
                lagint(lagpol, order=1, k=[i], lbnd=-1),
            ),
            i,
        )

    for i in range(5):
        scl = i + 1
        pol = tensor([0] * i + [1])
        target = [i] + [0] * i + [2 / scl]
        lagpol = poly2lag(pol)
        assert_close(
            lagtrim(lag2poly(lagint(lagpol, order=1, k=[i], scl=2)), tol=0.000001),
            lagtrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for _ in range(j):
                target = lagint(target, order=1)
            assert_close(
                lagtrim(lagint(pol, order=j), tol=0.000001),
                lagtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = lagint(target, order=1, k=[k])
            assert_close(
                lagtrim(lagint(pol, order=j, k=list(range(j))), tol=0.000001),
                lagtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = lagint(target, order=1, k=[k], lbnd=-1)
            assert_close(
                lagtrim(lagint(pol, order=j, k=list(range(j)), lbnd=-1), tol=0.000001),
                lagtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = lagint(target, order=1, k=[k], scl=2)
            assert_close(
                lagtrim(
                    lagint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                lagtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([lagint(c) for c in c2d.T]).T
    assert_close(
        lagint(c2d, axis=0),
        target,
    )

    target = vstack([lagint(c) for c in c2d])
    res = lagint(
        c2d,
        axis=1,
    )
    assert_close(
        res,
        target,
    )

    target = vstack([lagint(c, k=3) for c in c2d])
    res = lagint(
        c2d,
        k=3,
        axis=1,
    )
    assert_close(
        res,
        target,
    )


def test_lagline():
    assert_close(lagline(3, 4), [7, -4])


def test_lagmul():
    x = linspace(-3, 3, 100)

    for i in range(5):
        pol1 = tensor([0] * i + [1])
        val1 = lagval(x, pol1)
        for j in range(5):
            pol2 = tensor([0] * j + [1])
            val2 = lagval(x, pol2)
            pol3 = lagtrim(lagmul(pol1, pol2))
            val3 = lagval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_close(val3, val1 * val2)


def test_lagmulx():
    assert_close(
        lagtrim(
            lagmulx(
                [0],
            ),
            tol=0.000001,
        ),
        tensor([0]),
    )

    assert_close(
        lagtrim(
            lagmulx(
                tensor([1]),
            ),
            tol=0.000001,
        ),
        tensor([1, -1]),
    )

    for i in range(1, 5):
        assert_close(
            lagtrim(
                lagmulx(
                    tensor([0] * i + [1]),
                ),
                tol=0.000001,
            ),
            lagtrim(
                tensor([0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]),
                tol=0.000001,
            ),
        )


def test_lagone():
    assert_close(
        lagone,
        tensor([1]),
    )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)
            assert_close(
                lagtrim(lagpow(c, j), tol=0.000001),
                lagtrim(
                    functools.reduce(lagmul, [c] * j, tensor([1])),
                    tol=0.000001,
                ),
            )


def test_lagroots():
    assert_close(lagroots([1]), tensor([]))
    assert_close(
        lagroots([0, 1]),
        tensor([1]),
    )
    for i in range(2, 5):
        assert_close(
            lagtrim(
                lagroots(lagfromroots(linspace(0, 3, i))),
                tol=0.000001,
            ),
            lagtrim(linspace(0, 3, i), tol=0.000001),
        )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] - 1)

            assert_close(
                lagtrim(
                    lagsub(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_lagtrim():
    pytest.raises(ValueError, lagtrim, tensor([2, -1, 1, 0]), -1)

    assert_close(lagtrim(tensor([2, -1, 1, 0])), tensor([2, -1, 1, 0])[:-1])
    assert_close(lagtrim(tensor([2, -1, 1, 0]), 1), tensor([2, -1, 1, 0])[:-3])
    assert_close(
        lagtrim(tensor([2, -1, 1, 0]), 2),
        tensor([0]),
    )


def test_lagval():
    assert_close(lagval(tensor([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [polyval(x, c) for c in lagcoefficients]
    for i in range(7):
        assert_close(
            lagval(x, tensor([0] * i + [1])),
            y[i],
        )

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_close(lagval(x, [1]).shape, dims)
        assert_close(lagval(x, tensor([1, 0])).shape, dims)
        assert_close(lagval(x, tensor([1, 0, 0])).shape, dims)


def test_lagval2d():
    c1d = tensor([9.0, -14.0, 6.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

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
    assert_close(
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
    c1d = tensor([9.0, -14.0, 6.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    pytest.raises(ValueError, lagval3d, x1, x2, x3[:2], c3d)

    assert_close(
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
        assert_close(
            v[..., i],
            lagval(
                x,
                tensor([0] * i + [1]),
            ),
        )

    x = tensor([[1, 2], [3, 4], [5, 6]])

    v = lagvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_close(
            v[..., i],
            lagval(
                x,
                tensor([0] * i + [1]),
            ),
        )


def test_lagvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_close(
        dot(lagvander2d(x1, x2, (1, 2)), c.ravel()),
        lagval2d(x1, x2, c),
    )

    assert lagvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_lagvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_close(
        dot(lagvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        lagval3d(x1, x2, x3, c),
    )

    assert lagvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_lagweight():
    assert_close(
        lagweight(linspace(0, 10, 11)),
        exp(-linspace(0, 10, 11)),
    )


def test_lagx():
    assert_close(lagx, [1, -1])


def test_lagzero():
    assert_close(
        lagzero,
        tensor([0]),
    )


def test_leg2poly():
    for index in range(10):
        assert_close(
            leg2poly([0] * index + [1]),
            legcoefficients[index],
        )


def test_legadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_close(
                legtrim(
                    legadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
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
        legcompanion(tensor([]))

    with pytest.raises(ValueError):
        legcompanion(tensor([1]))

    for i in range(1, 5):
        coef = tensor([0] * i + [1])
        assert legcompanion(coef).shape == (i, i)

    assert legcompanion(tensor([1, 2]))[0, 0] == -0.5


def test_legder():
    pytest.raises(TypeError, legder, tensor([0]), 0.5)
    pytest.raises(ValueError, legder, tensor([0]), -1)

    for i in range(5):
        target = tensor([0] * i + [1])
        res = legder(target, order=0)
        assert_close(
            legtrim(res, tol=0.000001),
            legtrim(target, tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            target = tensor([0] * i + [1])
            res = legder(legint(target, order=j), order=j)
            assert_close(
                legtrim(res, tol=0.000001),
                legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            target = tensor([0] * i + [1])
            res = legder(legint(target, order=j, scl=2), order=j, scl=0.5)
            assert_close(
                legtrim(res, tol=0.000001),
                legtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    target = vstack([legder(c) for c in c2d.T]).T
    res = legder(c2d, axis=0)
    assert_close(
        res,
        target,
    )

    target = vstack([legder(c) for c in c2d])
    res = legder(
        c2d,
        axis=1,
    )
    assert_close(
        res,
        target,
    )

    c = (1, 2, 3, 4)
    assert_close(
        legder(c, 4),
        tensor([0]),
    )


def test_legdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = legdiv(
                legadd(
                    tensor([0] * i + [1]),
                    tensor([0] * j + [1]),
                ),
                tensor([0] * i + [1]),
            )

            assert_close(
                legtrim(
                    legadd(
                        legmul(
                            quotient,
                            tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    legadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legdomain():
    assert_close(legdomain, [-1, 1])


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    coef3 = legfit(linspace(0, 2), f(linspace(0, 2)), 3)
    assert_close(len(coef3), 4)
    assert_close(
        legval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )
    coef3 = legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3),
    )
    assert_close(
        len(coef3),
        4,
    )
    assert_close(
        legval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )

    coef4 = legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        4,
    )
    assert_close(
        len(coef4),
        5,
    )
    assert_close(
        legval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )
    coef4 = legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3, 4),
    )
    assert_close(
        len(coef4),
        5,
    )
    assert_close(
        legval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )

    coef4 = legfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (2, 3, 4, 1, 0),
    )

    assert_close(
        len(coef4),
        5,
    )

    assert_close(
        legval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )

    assert_close(
        legfit(
            linspace(0, 2),
            tensor([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
            3,
        ),
        tensor([coef3, coef3]).T,
    )

    assert_close(
        legfit(
            linspace(0, 2),
            tensor([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
            (0, 1, 2, 3),
        ),
        tensor([coef3, coef3]).T,
    )

    w = zeros_like(linspace(0, 2))

    yw = f(linspace(0, 2)).copy()

    w = w.at[1::2].set(1)

    i = f(linspace(0, 2))

    i = i.at[0::2].set(0)

    assert_close(
        legfit(
            linspace(0, 2),
            yw,
            3,
            weight=w,
        ),
        coef3,
    )

    assert_close(
        legfit(
            linspace(0, 2),
            yw,
            (0, 1, 2, 3),
            weight=w,
        ),
        coef3,
    )

    assert_close(
        legfit(
            linspace(0, 2),
            tensor([yw, yw]).T,
            3,
            weight=w,
        ),
        tensor([coef3, coef3]).T,
    )

    assert_close(
        legfit(
            linspace(0, 2),
            tensor([yw, yw]).T,
            (0, 1, 2, 3),
            weight=w,
        ),
        tensor([coef3, coef3]).T,
    )

    assert_close(
        legfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            1,
        ),
        [0, 1],
    )

    assert_close(
        legfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            (0, 1),
        ),
        [0, 1],
    )

    assert_close(
        legval(
            linspace(-1, 1),
            legfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                4,
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_close(
        legval(
            linspace(-1, 1),
            legfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                (0, 2, 4),
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_close(
        legfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            4,
        ),
        legfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            (0, 2, 4),
        ),
    )


def test_legfromroots():
    assert_close(
        legtrim(legfromroots(tensor([])), tol=0.000001),
        [1],
    )
    for i in range(1, 5):
        assert (
            legfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])).shape[-1] == i + 1
        )
        assert_close(
            leg2poly(legfromroots(cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])))[-1],
            1,
        )
        assert_close(
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

    assert_close(
        vv,
        eye(100),
    )

    assert_close(w.sum(), 2.0)


def test_leggrid2d():
    c1d = tensor([2.0, 2.0, 2.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
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
    c1d = tensor([2.0, 2.0, 2.0])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
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
    pytest.raises(TypeError, legint, tensor([0]), 0.5)
    pytest.raises(ValueError, legint, tensor([0]), -1)
    pytest.raises(
        ValueError,
        legint,
        tensor([0]),
        1,
        tensor([0, 0]),
    )
    pytest.raises(ValueError, legint, tensor([0]), lbnd=[0])
    pytest.raises(ValueError, legint, tensor([0]), scl=[0])
    pytest.raises(TypeError, legint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        assert_close(
            legtrim(
                legint(tensor([0]), order=i, k=([0] * (i - 2) + [1])),
                tol=0.000001,
            ),
            [0, 1],
        )

    for i in range(5):
        assert_close(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(tensor([0] * i + [1])),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            legtrim(tensor([i] + [0] * i + [1 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        assert_close(
            legval(
                tensor([-1]),
                legint(
                    poly2leg(tensor([0] * i + [1])),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_close(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(tensor([0] * i + [1])),
                        order=1,
                        k=[i],
                        scl=2,
                    )
                ),
                tol=0.000001,
            ),
            legtrim(tensor([i] + [0] * i + [2 / (i + 1)]), tol=0.000001),
        )

    for i in range(5):
        for j in range(2, 5):
            target = (tensor([0] * i + [1]))[:]
            for _ in range(j):
                target = legint(target, order=1)
            assert_close(
                legtrim(legint(tensor([0] * i + [1]), order=j), tol=0.000001),
                legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = legint(target, order=1, k=[k])
            assert_close(
                legtrim(legint(pol, order=j, k=list(range(j))), tol=0.000001),
                legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = legint(target, order=1, k=[k], lbnd=-1)
            assert_close(
                legtrim(
                    legint(pol, order=j, k=list(range(j)), lbnd=-1),
                    tol=0.000001,
                ),
                legtrim(target, tol=0.000001),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])
            target = pol[:]
            for k in range(j):
                target = legint(target, order=1, k=[k], scl=2)
            assert_close(
                legtrim(
                    legint(pol, order=j, k=list(range(j)), scl=2),
                    tol=0.000001,
                ),
                legtrim(target, tol=0.000001),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_close(
        legint(c2d, axis=0),
        vstack([legint(c) for c in c2d.T]).T,
    )

    assert_close(
        legint(
            c2d,
            axis=1,
        ),
        vstack([legint(c) for c in c2d]),
    )

    assert_close(
        legint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([legint(c, k=3) for c in c2d]),
    )

    assert_close(legint((1, 2, 3), 0), (1, 2, 3))


def test_legline():
    assert_close(legline(3, 4), tensor([3, 4]))

    assert_close(
        legtrim(
            legline(3, 0),
            tol=0.000001,
        ),
        [3],
    )


def test_legmul():
    for i in range(5):
        pol1 = tensor([0] * i + [1])
        x = linspace(-1, 1, 100)
        val1 = legval(x, pol1)
        for j in range(5):
            pol2 = tensor([0] * j + [1])
            val2 = legval(x, pol2)
            pol3 = legmul(pol1, pol2)
            val3 = legval(x, pol3)
            assert len(pol3) == i + j + 1
            assert_close(val3, val1 * val2)


def test_legmulx():
    assert_close(
        legtrim(
            legmulx(
                tensor([0]),
            ),
            tol=0.000001,
        ),
        tensor([0]),
    )

    assert_close(
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

        assert_close(
            legtrim(
                legmulx(
                    [0] * index + [1],
                ),
                tol=0.000001,
            ),
            [0] * (index - 1) + [index / tmp, 0, (index + 1) / tmp],
        )


def test_legone():
    assert_close(
        legone,
        tensor([1]),
    )


def test_legpow():
    for i in range(5):
        for j in range(5):
            c = arange(i + 1)

            assert_close(
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
                        tensor([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legroots():
    assert_close(legroots([1]), tensor([]))
    assert_close(legroots(tensor([1, 2])), tensor([-0.5]))

    for index in range(2, 5):
        assert_close(
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

            assert_close(
                legtrim(
                    legsub(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_legtrim():
    pytest.raises(ValueError, legtrim, tensor([2, -1, 1, 0]), -1)

    assert_close(legtrim(tensor([2, -1, 1, 0])), tensor([2, -1, 1, 0])[:-1])
    assert_close(legtrim(tensor([2, -1, 1, 0]), 1), tensor([2, -1, 1, 0])[:-3])
    assert_close(
        legtrim(tensor([2, -1, 1, 0]), 2),
        tensor([0]),
    )


def test_legval():
    assert_close(legval(tensor([]), [1]).size, 0)

    x = linspace(-1, 1)
    y = [polyval(x, c) for c in legcoefficients]
    for i in range(10):
        assert_close(legval(x, tensor([0] * i + [1])), y[i])

    for i in range(3):
        dims = [2] * i
        x = zeros(dims)
        assert_close(legval(x, [1]).shape, dims)
        assert_close(legval(x, tensor([1, 0])).shape, dims)
        assert_close(legval(x, tensor([1, 0, 0])).shape, dims)


def test_legval2d():
    c1d = tensor([2.0, 2.0, 2.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        legval2d,
        x1,
        x2[:2],
        c2d,
    )

    assert_close(
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
    c1d = tensor([2.0, 2.0, 2.0])
    c3d = einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    x1, x2, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, legval3d, x1, x2, x3[:2], c3d)

    assert_close(
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
        assert_close(
            v[..., index],
            legval(
                x,
                [0] * index + [1],
            ),
        )

    x = tensor([[1, 2], [3, 4], [5, 6]])
    v = legvander(x, 3)
    assert v.shape == (3, 2, 4)

    for index in range(4):
        assert_close(
            v[..., index],
            legval(
                x,
                [0] * index + [1],
            ),
        )

    pytest.raises(ValueError, legvander, (1, 2, 3), -1)


def test_legvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3))
    assert_close(
        dot(legvander2d(x1, x2, (1, 2)), c.ravel()),
        legval2d(x1, x2, c),
    )

    assert legvander2d([x1], [x2], (1, 2)).shape == (1, 5, 6)


def test_legvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    c = jax.random.uniform(key, (2, 3, 4))
    assert_close(
        dot(legvander3d(x1, x2, x3, (1, 2, 3)), c.ravel()),
        legval3d(x1, x2, x3, c),
    )

    assert legvander3d([x1], [x2], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_legweight():
    assert_close(legweight(linspace(-1, 1, 11)), 1.0)


def test_legx():
    assert_close(
        legx,
        tensor([0, 1]),
    )


def test_legzero():
    assert_close(
        legzero,
        tensor([0]),
    )


def test_poly2cheb():
    for i in range(10):
        assert_close(
            poly2cheb(
                chebcoefficients[i],
            ),
            tensor([0] * i + [1]),
        )


def test_poly2herm():
    for i in range(10):
        assert_close(
            hermtrim(
                poly2herm(
                    hermcoefficients[i],
                ),
                tol=0.000001,
            ),
            tensor([0] * i + [1]),
        )


def test_poly2herme():
    for i in range(10):
        assert_close(
            poly2herme(
                hermecoefficients[i],
            ),
            tensor([0] * i + [1]),
        )


def test_poly2lag():
    for i in range(7):
        assert_close(
            poly2lag(
                lagcoefficients[i],
            ),
            tensor([0] * i + [1]),
        )


def test_poly2leg():
    for i in range(10):
        assert_close(
            poly2leg(
                legcoefficients[i],
            ),
            tensor([0] * i + [1]),
        )


def test_polyadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target = target.at[i].set(target[i] + 1)
            target = target.at[j].set(target[j] + 1)

            assert_close(
                polytrim(
                    polyadd(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
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
        polycompanion(tensor([]))

    with pytest.raises(ValueError):
        polycompanion(tensor([1]))

    for i in range(1, 5):
        output = polycompanion(
            tensor([0] * i + [1]),
        )

        assert output.shape == (i, i)

    output = polycompanion(
        tensor([1, 2]),
    )

    assert output[0, 0] == -0.5


def test_polyder():
    with pytest.raises(TypeError):
        polyder(tensor([0]), axis=0.5)

    for i in range(5):
        assert_close(
            polytrim(
                polyder(
                    tensor([0] * i + [1]),
                    order=0,
                ),
                tol=0.000001,
            ),
            polytrim(
                tensor([0] * i + [1]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                polytrim(
                    polyder(
                        polyint(
                            tensor([0] * i + [1]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    tensor([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                polytrim(
                    polyder(
                        polyint(
                            tensor([0] * i + [1]),
                            order=j,
                            scl=2,
                        ),
                        order=j,
                        scl=0.5,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    tensor([0] * i + [1]),
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 4))

    assert_close(
        polyder(
            c2d,
            axis=0,
        ),
        vstack([polyder(c) for c in c2d.T]).T,
    )

    assert_close(
        polyder(
            c2d,
            axis=1,
        ),
        vstack([polyder(c) for c in c2d]),
    )


def test_polydiv():
    quotient, remainder = polydiv(
        tensor([2]),
        tensor([2]),
    )

    assert_close(
        quotient,
        tensor([1]),
    )

    assert_close(
        remainder,
        tensor([0]),
    )

    quotient, remainder = polydiv(
        tensor([2, 2]),
        tensor([2]),
    )

    assert_close(
        quotient,
        tensor([1, 1]),
    )

    assert_close(
        remainder,
        tensor([0]),
    )

    for i in range(5):
        for j in range(5):
            target = polyadd(
                tensor([0.0] * i + [1.0, 2.0]),
                tensor([0.0] * j + [1.0, 2.0]),
            )

            quotient, remainder = polydiv(
                target,
                tensor([0.0] * i + [1.0, 2.0]),
            )

            assert_close(
                polytrim(
                    polyadd(
                        polymul(
                            quotient,
                            tensor([0.0] * i + [1.0, 2.0]),
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
    assert_close(
        polydomain,
        tensor([-1, 1]),
    )


def test_polyfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    coef3 = polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        3,
    )
    assert coef3.shape[0] == 4
    assert_close(
        polyval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )
    coef3 = polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3),
    )
    assert len(coef3) == 4
    assert_close(
        polyval(linspace(0, 2), coef3),
        f(linspace(0, 2)),
    )

    coef4 = polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        4,
    )
    assert len(coef4) == 5
    assert_close(
        polyval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )
    coef4 = polyfit(
        linspace(0, 2),
        f(linspace(0, 2)),
        (0, 1, 2, 3, 4),
    )
    assert len(coef4) == 5
    assert_close(
        polyval(linspace(0, 2), coef4),
        f(linspace(0, 2)),
    )

    coef2d = polyfit(
        linspace(0, 2),
        tensor([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
        3,
    )
    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )
    coef2d = polyfit(
        linspace(0, 2),
        tensor([(f(linspace(0, 2))), (f(linspace(0, 2)))]).T,
        (0, 1, 2, 3),
    )
    assert_close(
        coef2d,
        tensor([coef3, coef3]).T,
    )

    w = zeros_like(linspace(0, 2))

    w = w.at[1::2].set(1)
    yw = f(linspace(0, 2)).at[0::2].set(0)

    wcoef3 = polyfit(
        linspace(0, 2),
        yw,
        3,
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    wcoef3 = polyfit(
        linspace(0, 2),
        yw,
        (0, 1, 2, 3),
        weight=w,
    )

    assert_close(
        wcoef3,
        coef3,
    )

    assert_close(
        polyfit(
            linspace(0, 2),
            tensor([yw, yw]).T,
            3,
            weight=w,
        ),
        tensor([coef3, coef3]).T,
    )

    assert_close(
        polyfit(
            linspace(0, 2),
            tensor([yw, yw]).T,
            (0, 1, 2, 3),
            weight=w,
        ),
        tensor([coef3, coef3]).T,
    )

    assert_close(
        polyfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            1,
        ),
        tensor([0, 1]),
    )

    assert_close(
        polyfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            (0, 1),
        ),
        tensor([0, 1]),
    )

    assert_close(
        polyval(
            linspace(-1, 1),
            polyfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                4,
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_close(
        polyval(
            linspace(-1, 1),
            polyfit(
                linspace(-1, 1),
                g(linspace(-1, 1)),
                (0, 2, 4),
            ),
        ),
        g(linspace(-1, 1)),
    )

    assert_close(
        polyfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            4,
        ),
        polyfit(
            linspace(-1, 1),
            g(linspace(-1, 1)),
            (0, 2, 4),
        ),
    )


def test_polyfromroots():
    assert_close(
        polytrim(
            polyfromroots(tensor([])),
            tol=0.000001,
        ),
        tensor([1]),
    )

    for i in range(1, 5):
        roots = cos(
            linspace(-math.pi, 0, 2 * i + 1)[1::2],
        )

        assert_close(
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
    y1, y2, y3 = polyval(x, [1.0, 2.0, 3.0])

    assert_close(
        polygrid2d(
            x1,
            x2,
            einsum(
                "i,j->ij",
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
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
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 2


def test_polygrid3d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    x1, x2, x3 = x
    y1, y2, y3 = y

    assert_close(
        polygrid3d(
            x1,
            x2,
            x3,
            einsum(
                "i,j,k->ijk",
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
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
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_polyint():
    with pytest.raises(TypeError):
        polyint(tensor([0]), 0.5)

    with pytest.raises(ValueError):
        polyint(tensor([0]), -1)

    with pytest.raises(ValueError):
        polyint(
            tensor([0]),
            1,
            tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        polyint(tensor([0]), lbnd=[0])

    with pytest.raises(ValueError):
        polyint(tensor([0]), scl=[0])

    with pytest.raises(TypeError):
        polyint(tensor([0]), axis=0.5)

    for i in range(2, 5):
        assert_close(
            polytrim(
                polyint(
                    tensor([0]),
                    order=i,
                    k=([0] * (i - 2) + [1]),
                ),
                tol=0.000001,
            ),
            [0, 1],
        )

    for i in range(5):
        assert_close(
            polytrim(
                polyint(
                    tensor([0] * i + [1]),
                    order=1,
                    k=[i],
                ),
                tol=0.000001,
            ),
            polytrim(
                tensor([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_close(
            polyval(
                tensor([-1]),
                polyint(
                    tensor([0] * i + [1]),
                    order=1,
                    k=[i],
                    lbnd=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_close(
            polytrim(
                polyint(
                    tensor([0] * i + [1]),
                    order=1,
                    k=[i],
                    scl=2,
                ),
                tol=0.000001,
            ),
            polytrim(
                tensor([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0] * i + [1])

            target = pol[:]

            for _ in range(j):
                target = polyint(
                    target,
                    order=1,
                )

            assert_close(
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
            pol = tensor([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                )

            assert_close(
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
            pol = tensor([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                    lbnd=-1,
                )

            assert_close(
                polytrim(
                    polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lbnd=-1,
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
            pol = tensor([0] * i + [1])

            target = pol[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                    scl=2,
                )

            assert_close(
                polytrim(
                    polyint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scl=2,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = jax.random.uniform(key, (3, 6))

    assert_close(
        polyint(
            c2d,
            axis=0,
        ),
        vstack([polyint(c) for c in c2d.T]).T,
    )

    assert_close(
        polyint(
            c2d,
            axis=1,
        ),
        vstack([polyint(c) for c in c2d]),
    )

    assert_close(
        polyint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([polyint(c, k=3) for c in c2d]),
    )


def test_polyline():
    assert_close(
        polyline(3, 4),
        tensor([3, 4]),
    )

    assert_close(
        polyline(3, 0),
        tensor([3, 0]),
    )


def test_polymul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[i + j].set(target[i + j] + 1)

            assert_close(
                polytrim(
                    polymul(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polymulx():
    assert_close(
        polymulx(
            tensor([0]),
        ),
        tensor([0, 0]),
    )

    assert_close(
        polymulx(
            tensor([1]),
        ),
        tensor([0, 1]),
    )

    for i in range(1, 5):
        assert_close(
            polymulx(
                tensor([0] * i + [1]),
            ),
            tensor([0] * (i + 1) + [1]),
        )


def test_polyone():
    assert_close(
        polyone,
        tensor([1]),
    )


def test_polypow():
    for i in range(5):
        for j in range(5):
            assert_close(
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
                        [(arange(i + 1))] * j,
                        tensor([1]),
                    ),
                    tol=0.000001,
                ),
            )


def test_polyroots():
    assert_close(
        polyroots([1]),
        tensor([]),
    )

    assert_close(
        polyroots(tensor([1, 2])),
        tensor([-0.5]),
    )

    for i in range(2, 5):
        assert_close(
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

            assert_close(
                polytrim(
                    polysub(
                        tensor([0] * i + [1]),
                        tensor([0] * j + [1]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polytrim():
    coef = tensor([2, -1, 1, 0])

    pytest.raises(ValueError, polytrim, coef, -1)

    assert_close(
        polytrim(coef),
        coef[:-1],
    )

    assert_close(
        polytrim(coef, 1),
        coef[:-3],
    )

    assert_close(
        polytrim(coef, 2),
        tensor([0]),
    )


def test_polyval():
    assert polyval(tensor([]), [1]).size == 0

    x = linspace(-1, 1)
    y = [x**i for i in range(5)]

    for i in range(5):
        assert_close(
            polyval(x, tensor([0] * i + [1])),
            y[i],
        )

    assert_close(
        polyval(x, [0, -1, 0, 1]),
        x * (x**2 - 1),
    )

    for i in range(3):
        dims = (2,) * i
        x = zeros(dims)

        assert polyval(x, tensor([1])).shape == dims

        assert polyval(x, tensor([1, 0])).shape == dims

        assert polyval(x, tensor([1, 0, 0])).shape == dims


def test_polyval2d():
    x = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    x1, x2, x3 = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        polyval2d(
            x1,
            x2,
            einsum(
                "i,j->ij",
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
            ),
        ),
        y1 * y2,
    )

    output = polyval2d(
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j->ij",
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
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
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        polyval3d(
            x1,
            x2,
            x3,
            einsum(
                "i,j,k->ijk",
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
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
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_polyvalfromroots():
    pytest.raises(
        ValueError,
        polyvalfromroots,
        tensor([1]),
        tensor([1]),
        tensor=False,
    )

    assert polyvalfromroots(tensor([]), tensor([1])).size == 0

    assert polyvalfromroots(tensor([]), tensor([1])).shape == (0,)

    assert polyvalfromroots(tensor([]), [[1] * 5]).size == 0

    assert polyvalfromroots(tensor([]), [[1] * 5]).shape == (5, 0)

    assert_close(
        polyvalfromroots(
            tensor([1]),
            tensor([1]),
        ),
        tensor([0]),
    )

    assert polyvalfromroots(tensor([1]), ones((3, 3))).shape == (3, 1)

    y = [linspace(-1, 1) ** i for i in range(5)]
    for i in range(1, 5):
        target = y[i]
        res = polyvalfromroots(linspace(-1, 1), [0] * i)
        assert_close(
            res,
            target,
        )
    target = linspace(-1, 1) * (linspace(-1, 1) - 1) * (linspace(-1, 1) + 1)
    res = polyvalfromroots(linspace(-1, 1), tensor([-1, 0, 1]))
    assert_close(
        res,
        target,
    )

    for i in range(3):
        dims = (2,) * i
        x = zeros(dims)
        assert polyvalfromroots(x, tensor([1])).shape == dims
        assert polyvalfromroots(x, tensor([1, 0])).shape == dims
        assert polyvalfromroots(x, tensor([1, 0, 0])).shape == dims

    ptest = tensor([15, 2, -16, -2, 1])
    r = polyroots(ptest)
    assert_close(
        polyval(linspace(-1, 1), ptest),
        polyvalfromroots(linspace(-1, 1), r),
    )

    rshape = (3, 5)

    x = arange(-3, 2)

    r = jax.random.randint(key, rshape, -5, 5)

    target = empty(r.shape[1:])

    for ii in range(target.size):
        target = target.at[ii].set(polyvalfromroots(x[ii], r[:, ii]))

    assert_close(
        polyvalfromroots(x, r, tensor=False),
        target,
    )

    x = vstack([x, 2 * x])

    target = empty(r.shape[1:] + x.shape)

    for ii in range(r.shape[1]):
        for jj in range(x.shape[0]):
            target = target.at[ii, jj, :].set(
                polyvalfromroots(
                    x[jj],
                    r[:, ii],
                )
            )

    assert_close(
        polyvalfromroots(x, r, tensor=True),
        target,
    )


def test_polyvander():
    x = arange(3)

    v = polyvander(x, 3)

    assert v.shape == (3, 4)

    for i in range(4):
        assert_close(
            v[..., i],
            polyval(
                x,
                tensor([0] * i + [1]),
            ),
        )

    x = tensor([[1, 2], [3, 4], [5, 6]])

    v = polyvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_close(
            v[..., i],
            polyval(
                x,
                tensor([0] * i + [1]),
            ),
        )

    pytest.raises(
        ValueError,
        polyvander,
        arange(3),
        -1,
    )


def test_polyvander2d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    c = jax.random.uniform(key, (2, 3))

    assert_close(
        dot(
            polyvander2d(
                x1,
                x2,
                (1, 2),
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
        (1, 2),
    )

    assert van.shape == (1, 5, 6)


def test_polyvander3d():
    x1, x2, x3 = jax.random.uniform(key, (3, 5), minval=-1, maxval=1)

    c = jax.random.uniform(key, (2, 3, 4))

    assert_close(
        dot(
            polyvander3d(
                x1,
                x2,
                x3,
                degree=(1, 2, 3),
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
        degree=(1, 2, 3),
    )

    assert van.shape == (1, 5, 24)


def test_polyx():
    assert_close(
        polyx,
        tensor([0, 1]),
    )


def test_polyzero():
    assert_close(
        polyzero,
        tensor([0]),
    )
