import functools
import math

import pytest
import torch
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
    polydiv,
    polydomain,
    polyfit,
    polyfromroots,
    polygrid2d,
    polygrid3d,
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
from torch import (
    Tensor,
    arange,
    cos,
    dot,
    einsum,
    empty,
    exp,
    eye,
    float64,
    linspace,
    ones,
    rand,
    randint,
    ravel,
    set_default_dtype,
    sqrt,
    tensor,
    vstack,
    zeros,
    zeros_like,
)
from torch.testing import (
    assert_close,
)

set_default_dtype(float64)

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
    for index in range(5):
        assert_close(
            _c_series_to_z_series(
                tensor([2.0] + [1.0] * index),
            ),
            tensor([0.5] * index + [2.0] + [0.5] * index),
        )


def test__fit():
    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0]),
            degree=tensor([-1.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([[1.0]]),
            tensor([1.0]),
            degree=tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([]),
            tensor([1.0]),
            degree=tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([[[1.0]]]),
            degree=tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([1.0, 2.0]),
            tensor([1.0]),
            degree=tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0, 2.0]),
            degree=tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0]),
            degree=tensor([0.0]),
            weight=[[1.0]],
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0]),
            degree=tensor([0.0]),
            weight=tensor([1.0, 1.0]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0]),
            degree=tensor([-1.0]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0]),
            degree=tensor([2.0, -1.0, 6.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            tensor([1.0]),
            tensor([1.0]),
            degree=tensor([]),
        )


def test__get_domain():
    assert_close(
        _get_domain(
            tensor([1.0, 10.0, 3.0, -1.0]),
        ),
        tensor([-1.0, 10.0]),
    )

    assert_close(
        _get_domain(
            tensor([1 + 1j, 1 - 1j, 0.0, 2.0]),
        ),
        tensor([-0 - 1j, 2 + 1j]),
    )


def test__map_domain():
    assert_close(
        _map_domain(
            tensor([0.0, 4.0]),
            tensor([0.0, 4.0]),
            tensor([1.0, 3.0]),
        ),
        tensor([1.0, 3.0]),
    )

    assert_close(
        _map_domain(
            tensor([-0 - 1j, 2 + 1j]),
            tensor([-0 - 1j, 2 + 1j]),
            tensor([-2.0, 2.0]),
        ),
        tensor([-2.0, 2.0]),
    )

    assert_close(
        _map_domain(
            tensor([[0.0, 4.0], [0.0, 4.0]]),
            tensor([0.0, 4.0]),
            tensor([1.0, 3.0]),
        ),
        tensor([[1.0, 3.0], [1.0, 3.0]]),
    )


def test__map_parameters():
    assert_close(
        _map_parameters(
            tensor([0.0, 4.0]),
            tensor([1.0, 3.0]),
        ),
        tensor([1.0, 0.5]),
    )

    assert_close(
        _map_parameters(
            tensor([-1j, 2 + 1j]),
            tensor([-2 + 0j, 2 + 0j]),
        ),
        tensor([-1 + 1j, 1 - 1j]),
    )


def test__pow():
    with pytest.raises(ValueError):
        _pow(
            (),
            tensor([1.0, 2.0, 3.0]),
            exponent=5,
            maximum_exponent=4,
        )


def test__trim_coefficients():
    with pytest.raises(ValueError):
        _trim_coefficients(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_close(
        _trim_coefficients(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        _trim_coefficients(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        _trim_coefficients(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        tensor([0.0]),
    )


def test__trim_sequence():
    for _ in range(5):
        assert_close(
            _trim_sequence(
                tensor([1.0] + [0.0] * 5),
            ),
            tensor([1.0]),
        )


def test__vandermonde():
    with pytest.raises(ValueError):
        _vandermonde(
            (),
            tensor([1.0, 2.0, 3.0]),
            tensor([90.0]),
        )

    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (),
            tensor([90.65]),
        )

    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (),
            tensor([]),
        )


def test__z_series_to_c_series():
    for index in range(5):
        assert_close(
            _z_series_to_c_series(
                tensor([0.5] * index + [2.0] + [0.5] * index),
            ),
            tensor([2.0] + [1.0] * index),
        )


def test_chebadd():
    for j in range(5):
        for k in range(5):
            target = zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            assert_close(
                chebtrim(
                    chebadd(
                        tensor([0.0] * j + [1.0]),
                        tensor([0.0] * k + [1.0]),
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
        chebcompanion(
            tensor([]),
        )

    with pytest.raises(ValueError):
        chebcompanion(
            tensor([1.0]),
        )

    for index in range(1, 5):
        output = chebcompanion(
            tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = chebcompanion(
        tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.5


def test_chebder():
    with pytest.raises(TypeError):
        chebder(
            tensor([0]),
            tensor([0.5]),
        )

    with pytest.raises(ValueError):
        chebder(
            tensor([0]),
            tensor([-1.0]),
        )

    for index in range(5):
        input = tensor([0.0] * index + [1.0])

        assert_close(
            chebtrim(
                chebder(
                    input,
                    order=0,
                ),
                tol=0.000001,
            ),
            chebtrim(
                input,
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            input = tensor([0.0] * i + [1.0])

            assert_close(
                chebtrim(
                    chebder(
                        chebint(
                            input,
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    input,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            input = tensor([0.0] * i + [1.0])

            assert_close(
                chebtrim(
                    chebder(
                        chebint(
                            input,
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    input,
                    tol=0.000001,
                ),
            )

    input = rand(3, 4)

    target = [chebder(c) for c in input.T]

    target = vstack(target).T

    assert_close(
        chebder(
            input,
            axis=0,
        ),
        target,
    )

    target = [chebder(c) for c in input]

    target = vstack(target)

    assert_close(
        chebder(
            input,
            axis=1,
        ),
        target,
    )


def test_chebdiv():
    for j in range(5):
        for k in range(5):
            input = tensor([0.0] * j + [1.0])
            other = tensor([0.0] * k + [1.0])

            quotient, remainder = chebdiv(
                chebadd(
                    input,
                    other,
                ),
                input,
            )

            assert_close(
                chebtrim(
                    chebadd(
                        chebmul(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    chebadd(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )


def test_chebdomain():
    assert_close(
        chebdomain,
        tensor([-1.0, 1.0]),
    )


def test_chebfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([3]),
            ),
        ),
        other,
    )

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    assert_close(
        chebfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
        ),
        tensor(
            [
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    assert_close(
        chebfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
        ),
        tensor(
            [
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight[1::2] = 1.0

    assert_close(
        chebfit(
            input,
            other,
            degree=tensor([3]),
            weight=weight,
        ),
        chebfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        chebfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        chebfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        chebfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
            weight=weight,
        ),
        tensor(
            [
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    assert_close(
        chebfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        tensor(
            [
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                chebfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ],
        ).T,
    )

    assert_close(
        chebfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([1]),
        ),
        tensor([0, 1]),
    )

    assert_close(
        chebfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([0, 1]),
        ),
        tensor([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    assert_close(
        chebfit(
            input,
            other,
            degree=tensor([4]),
        ),
        chebfit(
            input,
            other,
            degree=tensor([0, 2, 4]),
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
        tensor([1.0]),
    )

    for index in range(1, 5):
        input = chebfromroots(
            cos(linspace(-math.pi, 0.0, 2 * index + 1)[1::2]),
        )

        input = input * 2 ** (index - 1)

        assert_close(
            chebtrim(
                input,
                tol=0.000001,
            ),
            chebtrim(
                tensor([0.0] * index + [1.0]),
                tol=0.000001,
            ),
        )


def test_chebgauss():
    output, weight = chebgauss(100)

    vandermonde = chebvander(
        output,
        degree=tensor([99]),
    )

    u = (vandermonde.T * weight) @ vandermonde

    v = 1 / sqrt(u.diagonal())

    assert_close(
        v[:, None] * u * v,
        eye(100),
    )

    assert_close(
        weight.sum(),
        tensor(math.pi),
    )


def test_chebgrid2d():
    x = rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        chebgrid2d(
            a,
            b,
            einsum(
                "i,j->ij",
                tensor([2.5, 2.0, 1.5]),
                tensor([2.5, 2.0, 1.5]),
            ),
        ),
        einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    res = chebgrid2d(
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j->ij",
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert res.shape == (2, 3) * 2


def test_chebgrid3d():
    x = rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        chebgrid3d(
            a,
            b,
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

    output = chebgrid3d(
        ones([2, 3]),
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j,k->ijk",
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_chebint():
    with pytest.raises(TypeError):
        chebint(
            tensor([0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        chebint(
            tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        chebint(
            tensor([0]),
            order=1,
            k=[0, 0],
        )

    with pytest.raises(ValueError):
        chebint(
            tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        chebint(
            tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        chebint(
            tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]

        assert_close(
            chebtrim(
                chebint(
                    tensor([0]),
                    order=i,
                    k=k,
                ),
                tol=0.000001,
            ),
            tensor([0, 1]),
        )

    for i in range(5):
        assert_close(
            chebtrim(
                cheb2poly(
                    chebint(
                        poly2cheb(
                            tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    ),
                ),
                tol=0.000001,
            ),
            chebtrim(
                tensor([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_close(
            chebval(
                tensor([-1]),
                chebint(
                    poly2cheb(
                        tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_close(
            chebtrim(
                cheb2poly(
                    chebint(
                        poly2cheb(
                            tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    )
                ),
                tol=0.000001,
            ),
            chebtrim(
                tensor([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            input = tensor([0.0] * i + [1.0])
            target = input[:]

            for _ in range(j):
                target = chebint(
                    target,
                    order=1,
                )

            assert_close(
                chebtrim(
                    chebint(
                        input,
                        order=j,
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
            input = tensor([0.0] * i + [1.0])

            target = input[:]

            for k in range(j):
                target = chebint(
                    target,
                    order=1,
                    k=[k],
                )

            assert_close(
                chebtrim(
                    chebint(
                        input,
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
            input = tensor([0.0] * i + [1.0])

            target = input[:]

            for k in range(j):
                target = chebint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            assert_close(
                chebtrim(
                    chebint(
                        input,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
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
            input = tensor([0.0] * i + [1.0])

            target = input[:]

            for k in range(j):
                target = chebint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            assert_close(
                chebtrim(
                    chebint(
                        input,
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

    assert_close(
        chebint(
            c2d,
            axis=0,
        ),
        vstack([chebint(c) for c in c2d.T]).T,
    )

    assert_close(
        chebint(
            c2d,
            axis=1,
        ),
        vstack([chebint(c) for c in c2d]),
    )

    assert_close(
        chebint(
            c2d,
            k=3,
            axis=1,
        ),
        vstack([chebint(c, k=3) for c in c2d]),
    )


def test_chebinterpolate():
    def f(x):
        return x * (x - 1) * (x - 2)

    with pytest.raises(ValueError):
        chebinterpolate(f, -1)

    for i in range(1, 5):
        assert chebinterpolate(f, i).shape == (i + 1,)

    def powx(x, p):
        return x**p

    x = linspace(-1, 1, 10)

    for i in range(0, 10):
        for j in range(0, i + 1):
            c = chebinterpolate(
                powx,
                i,
                (j,),
            )

            assert_close(
                chebval(x, c),
                powx(x, j),
            )


def test_chebline():
    assert_close(
        chebline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )


def test_chebmul():
    for j in range(5):
        for k in range(5):
            target = zeros(j + k + 1)

            target[abs(j + k)] = target[abs(j + k)] + 0.5
            target[abs(j - k)] = target[abs(j - k)] + 0.5

            input = tensor([0.0] * j + [1.0])
            other = tensor([0.0] * k + [1.0])

            assert_close(
                chebtrim(
                    chebmul(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebmulx():
    assert_close(
        chebtrim(
            chebmulx(
                tensor([0.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0]),
    )

    assert_close(
        chebtrim(
            chebmulx(
                tensor([1.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0, 1.0]),
    )

    for index in range(1, 5):
        assert_close(
            chebtrim(
                chebmulx(
                    tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            tensor([0.0] * (index - 1) + [0.5, 0, 0.5]),
        )


def test_chebone():
    assert_close(
        chebone,
        tensor([1.0]),
    )


def test_chebpow():
    for j in range(5):
        for k in range(5):
            assert_close(
                chebtrim(
                    chebpow(
                        arange(0.0, j + 1),
                        k,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    functools.reduce(
                        chebmul,
                        [arange(0.0, j + 1)] * k,
                        tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebpts1():
    with pytest.raises(ValueError):
        chebpts1(0)

    assert_close(
        chebpts1(1),
        tensor([0.0]),
    )

    assert_close(
        chebpts1(2),
        tensor([-0.70710678118654746, 0.70710678118654746]),
    )

    assert_close(
        chebpts1(3),
        tensor([-0.86602540378443871, 0, 0.86602540378443871]),
    )

    assert_close(
        chebpts1(4),
        tensor([-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]),
    )


def test_chebpts2():
    with pytest.raises(ValueError):
        chebpts2(1.5)

    with pytest.raises(ValueError):
        chebpts2(1)

    assert_close(
        chebpts2(2),
        tensor([-1.0, 1.0]),
    )

    assert_close(
        chebpts2(3),
        tensor([-1.0, 0.0, 1.0]),
    )

    assert_close(
        chebpts2(4),
        tensor([-1.0, -0.5, 0.5, 1.0]),
    )

    assert_close(
        chebpts2(5),
        tensor([-1.0, -0.707106781187, 0, 0.707106781187, 1.0]),
    )


def test_chebroots():
    assert_close(
        chebroots(
            tensor([1.0]),
        ),
        tensor([]),
    )

    assert_close(
        chebroots(
            tensor([1.0, 2.0]),
        ),
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
    for j in range(5):
        for k in range(5):
            target = zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] - 1

            input = tensor([0.0] * j + [1.0])
            other = tensor([0.0] * k + [1.0])

            assert_close(
                chebtrim(
                    chebsub(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebtrim():
    with pytest.raises(ValueError):
        chebtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_close(
        chebtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        chebtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        chebtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        tensor([0.0]),
    )


def test_chebval():
    output = chebval(
        tensor([]),
        tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in chebcoefficients:
        ys = [
            *ys,
            polyval(
                linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for index in range(10):
        assert_close(
            chebval(
                linspace(-1, 1, 50),
                tensor([0.0] * index + [1.0]),
            ),
            tensor(ys[index]),
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = chebval(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = chebval(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = chebval(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_chebval2d():
    x = rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    pytest.raises(
        ValueError,
        chebval2d,
        a,
        b[:2],
        einsum(
            "i,j->ij",
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert_close(
        chebval2d(
            a,
            b,
            einsum(
                "i,j->ij",
                tensor([2.5, 2.0, 1.5]),
                tensor([2.5, 2.0, 1.5]),
            ),
        ),
        y1 * y2,
    )

    res = chebval2d(
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j->ij",
            tensor([2.5, 2.0, 1.5]),
            tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert res.shape == (2, 3)


def test_chebval3d():
    c3d = einsum(
        "i,j,k->ijk",
        tensor([2.5, 2.0, 1.5]),
        tensor([2.5, 2.0, 1.5]),
        tensor([2.5, 2.0, 1.5]),
    )

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    pytest.raises(ValueError, chebval3d, a, b, x3[:2], c3d)

    assert_close(
        chebval3d(
            a,
            b,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    output = chebval3d(
        ones([2, 3]),
        ones([2, 3]),
        ones([2, 3]),
        c3d,
    )

    assert output.shape == (2, 3)


def test_chebvander():
    v = chebvander(
        arange(3),
        degree=3,
    )

    assert v.shape == (3, 4)

    for i in range(4):
        assert_close(
            v[..., i],
            chebval(
                arange(3),
                tensor([0.0] * i + [1.0]),
            ),
        )

    v = chebvander(
        tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=3,
    )

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_close(
            v[..., i],
            chebval(
                tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                tensor([0.0] * i + [1.0]),
            ),
        )


def test_chebvander2d():
    a, b, x3 = rand(3, 5) * 2 - 1

    c = rand(2, 3)

    assert_close(
        dot(
            chebvander2d(
                a,
                b,
                degree=tensor([1, 2]),
            ),
            ravel(c),
        ),
        chebval2d(
            a,
            b,
            c,
        ),
    )

    van = chebvander2d(
        [a],
        [b],
        degree=tensor([1, 2]),
    )

    assert van.shape == (1, 5, 6)


def test_chebvander3d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3, 4)

    assert_close(
        dot(
            chebvander3d(
                a,
                b,
                c,
                degree=tensor([1, 2, 3]),
            ),
            ravel(coefficients),
        ),
        chebval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = chebvander3d(
        [a],
        [b],
        [c],
        degree=tensor([1, 2, 3]),
    )

    assert output.shape == (1, 5, 24)


def test_chebweight():
    x = linspace(-1, 1, 11)[1:-1]

    assert_close(
        chebweight(x),
        1.0 / (sqrt(1 + x) * sqrt(1 - x)),
    )


def test_chebx():
    assert_close(
        chebx,
        tensor([0.0, 1.0]),
    )


def test_chebzero():
    assert_close(
        chebzero,
        tensor([0.0]),
    )


def test_herm2poly():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 2]),
        tensor([-2.0, 0, 4]),
        tensor([0.0, -12, 0, 8]),
        tensor([12.0, 0, -48, 0, 16]),
        tensor([0.0, 120, 0, -160, 0, 32]),
        tensor([-120.0, 0, 720, 0, -480, 0, 64]),
        tensor([0.0, -1680, 0, 3360, 0, -1344, 0, 128]),
        tensor([1680.0, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        tensor([0.0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    for index in range(10):
        assert_close(
            herm2poly(
                tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_hermadd():
    for j in range(5):
        for k in range(5):
            target = zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            assert_close(
                hermtrim(
                    hermadd(
                        tensor([0.0] * j + [1.0]),
                        tensor([0.0] * k + [1.0]),
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
        hermcompanion(
            tensor([]),
        )

    with pytest.raises(ValueError):
        hermcompanion(
            tensor([1.0]),
        )

    for index in range(1, 5):
        output = hermcompanion(
            tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = hermcompanion(
        tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.25


def test_hermder():
    with pytest.raises(TypeError):
        hermder(tensor([0]), 0.5)

    with pytest.raises(ValueError):
        hermder(tensor([0]), -1)

    for i in range(5):
        assert_close(
            hermtrim(
                hermder(
                    tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            hermtrim(
                tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                hermtrim(
                    hermder(
                        hermint(
                            tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = tensor([0.0] * i + [1.0])
            res = hermder(
                hermint(
                    target,
                    order=j,
                    scale=2,
                ),
                order=j,
                scale=0.5,
            )
            assert_close(
                hermtrim(
                    res,
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

    assert_close(
        hermder(c2d, axis=0),
        vstack([hermder(c) for c in c2d.T]).T,
    )

    assert_close(
        hermder(
            c2d,
            axis=1,
        ),
        vstack([hermder(c) for c in c2d]),
    )


def test_hermdiv():
    for j in range(5):
        for k in range(5):
            input = tensor([0.0] * j + [1.0])
            other = tensor([0.0] * k + [1.0])

            quotient, remainder = hermdiv(
                hermadd(
                    input,
                    other,
                ),
                input,
            )

            assert_close(
                hermtrim(
                    hermadd(
                        hermmul(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    hermadd(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )


def test_hermdomain():
    assert_close(
        hermdomain,
        tensor([-1.0, 1.0]),
    )


def test_herme2poly():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 1]),
        tensor([-1.0, 0, 1]),
        tensor([0.0, -3, 0, 1]),
        tensor([3.0, 0, -6, 0, 1]),
        tensor([0.0, 15, 0, -10, 0, 1]),
        tensor([-15.0, 0, 45, 0, -15, 0, 1]),
        tensor([0.0, -105, 0, 105, 0, -21, 0, 1]),
        tensor([105.0, 0, -420, 0, 210, 0, -28, 0, 1]),
        tensor([0.0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    for index in range(10):
        assert_close(
            herme2poly(
                tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_hermeadd():
    for j in range(5):
        for k in range(5):
            target = zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            assert_close(
                hermetrim(
                    hermeadd(
                        tensor([0.0] * j + [1.0]),
                        tensor([0.0] * k + [1.0]),
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
        hermecompanion(
            tensor([1.0]),
        )

    for index in range(1, 5):
        output = hermecompanion(
            tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = hermecompanion(
        tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.5


def test_hermeder():
    pytest.raises(TypeError, hermeder, tensor([0]), 0.5)
    pytest.raises(ValueError, hermeder, tensor([0]), -1)

    for i in range(5):
        assert_close(
            hermetrim(
                hermeder(tensor([0.0] * i + [1.0]), order=0),
                tol=0.000001,
            ),
            hermetrim(
                tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                hermetrim(
                    hermeder(
                        hermeint(tensor([0.0] * i + [1.0]), order=j),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                hermetrim(
                    hermeder(
                        hermeint(
                            tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

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
    for j in range(5):
        for k in range(5):
            input = tensor([0.0] * j + [1.0])
            other = tensor([0.0] * k + [1.0])

            quotient, remainder = hermediv(
                hermeadd(
                    input,
                    other,
                ),
                input,
            )

            assert_close(
                hermetrim(
                    hermeadd(
                        hermemul(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    hermeadd(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )


def test_hermedomain():
    assert_close(
        hermedomain,
        tensor([-1.0, 1.0]),
    )


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([3]),
            ),
        ),
        other,
    )

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    assert_close(
        hermefit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
        ),
        tensor(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        hermefit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
        ),
        tensor(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight[1::2] = 1.0

    assert_close(
        hermefit(
            input,
            other,
            degree=tensor([3]),
            weight=weight,
        ),
        hermefit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        hermefit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        hermefit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        hermefit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
            weight=weight,
        ),
        tensor(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        hermefit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        tensor(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        hermefit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([1]),
        ),
        tensor([0, 1]),
    )

    assert_close(
        hermefit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([0, 1]),
        ),
        tensor([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    assert_close(
        hermefit(
            input,
            other,
            degree=tensor([4]),
        ),
        hermefit(
            input,
            other,
            degree=tensor([0, 2, 4]),
        ),
    )


def test_hermefromroots():
    assert_close(
        hermetrim(
            hermefromroots(
                tensor([]),
            ),
            tol=0.000001,
        ),
        tensor([1.0]),
    )

    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])

        pol = hermefromroots(roots)

        assert len(pol) == i + 1

        assert_close(
            herme2poly(pol)[-1],
            tensor([1.0]),
        )

        assert_close(
            hermeval(roots, pol),
            tensor([0.0]),
        )


def test_hermegauss():
    x, w = hermegauss(100)

    v = hermevander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_close(vv, eye(100))

    target = math.sqrt(2 * math.pi)
    assert_close(
        w.sum(),
        tensor(target),
    )


def test_hermegrid2d():
    c1d = tensor([4.0, 2.0, 3.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j->ij", y1, y2)
    res = hermegrid2d(
        a,
        b,
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

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    assert_close(
        hermegrid3d(a, b, x3, c3d),
        einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = ones([2, 3])
    res = hermegrid3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3


def test_hermeint():
    pytest.raises(TypeError, hermeint, tensor([0]), 0.5)
    pytest.raises(ValueError, hermeint, tensor([0]), -1)
    pytest.raises(ValueError, hermeint, tensor([0]), 1, [0, 0])
    pytest.raises(ValueError, hermeint, tensor([0]), lower_bound=[0])
    pytest.raises(ValueError, hermeint, tensor([0]), scale=[0])
    pytest.raises(TypeError, hermeint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = hermeint(tensor([0]), order=i, k=k)
        assert_close(
            hermetrim(
                res,
                tol=0.000001,
            ),
            tensor([0, 1]),
        )

    for i in range(5):
        scale = i + 1
        pol = tensor([0.0] * i + [1.0])
        target = [i] + [0] * i + [1 / scale]
        hermepol = poly2herme(pol)
        res = herme2poly(hermeint(hermepol, order=1, k=[i]))
        assert_close(
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
        scale = i + 1
        pol = tensor([0.0] * i + [1.0])
        hermepol = poly2herme(pol)
        assert_close(
            hermeval(
                tensor([-1]),
                hermeint(
                    hermepol,
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        scale = i + 1
        pol = tensor([0.0] * i + [1.0])
        target = [i] + [0] * i + [2 / scale]
        hermepol = poly2herme(pol)
        res = herme2poly(hermeint(hermepol, order=1, k=[i], scale=2))
        assert_close(
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for _ in range(j):
                target = hermeint(target, order=1)
            res = hermeint(pol, order=j)
            assert_close(
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = hermeint(target, order=1, k=[k])

            assert_close(
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = hermeint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )
            assert_close(
                hermetrim(
                    hermeint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = hermeint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )
            assert_close(
                hermetrim(
                    hermeint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

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
    assert_close(
        hermeline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )


def test_hermemul():
    for index in range(5):
        input = linspace(-3, 3, 100)

        val1 = hermeval(
            input,
            tensor([0.0] * index + [1.0]),
        )

        for k in range(5):
            val2 = hermeval(
                input,
                tensor([0.0] * k + [1.0]),
            )

            assert_close(
                hermeval(
                    input,
                    hermemul(
                        tensor([0.0] * index + [1.0]),
                        tensor([0.0] * k + [1.0]),
                    ),
                ),
                val1 * val2,
            )


def test_hermemulx():
    assert_close(
        hermetrim(
            hermemulx(
                tensor([0.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0]),
    )
    assert_close(
        hermetrim(
            hermemulx(
                tensor([1.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0, 1.0]),
    )

    for index in range(1, 5):
        assert_close(
            hermetrim(
                hermemulx(
                    tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            tensor([0] * (index - 1) + [index, 0.0, 1.0]),
        )


def test_hermeone():
    assert_close(
        hermeone,
        tensor([1.0]),
    )


def test_hermepow():
    for j in range(5):
        for k in range(5):
            assert_close(
                hermetrim(
                    hermepow(
                        arange(0.0, j + 1),
                        k,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    functools.reduce(
                        hermemul,
                        [arange(0.0, j + 1)] * k,
                        tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermeroots():
    assert_close(
        hermeroots(
            tensor([1.0]),
        ),
        tensor([]),
    )

    assert_close(
        hermeroots(
            tensor([1.0, 1.0]),
        ),
        tensor([-1.0]),
    )

    for index in range(2, 5):
        input = linspace(-1, 1, index)

        assert_close(
            hermetrim(
                hermeroots(
                    hermefromroots(
                        input,
                    )
                ),
                tol=0.000001,
            ),
            hermetrim(
                input,
                tol=0.000001,
            ),
        )


def test_hermesub():
    for j in range(5):
        for k in range(5):
            target = zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] - 1

            input = tensor([0.0] * j + [1.0])
            other = tensor([0.0] * k + [1.0])

            assert_close(
                hermetrim(
                    hermesub(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermetrim():
    with pytest.raises(ValueError):
        hermetrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            -1,
        )

    assert_close(
        hermetrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        hermetrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        hermetrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            2,
        ),
        tensor([0.0]),
    )


def test_hermeval():
    output = hermeval(
        tensor([]),
        tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in hermecoefficients:
        ys = [
            *ys,
            polyval(
                linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for i in range(10):
        assert_close(
            hermeval(
                linspace(-1, 1, 50),
                tensor([0.0] * i + [1.0]),
            ),
            ys[i],
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = hermeval(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = hermeval(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = hermeval(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_hermeval2d():
    input = rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        tensor([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError):
        hermeval2d(
            a,
            b[:2],
            einsum(
                "i,j->ij",
                tensor([4.0, 2.0, 3.0]),
                tensor([4.0, 2.0, 3.0]),
            ),
        )

    assert_close(
        hermeval2d(
            a,
            b,
            einsum(
                "i,j->ij",
                tensor([4.0, 2.0, 3.0]),
                tensor([4.0, 2.0, 3.0]),
            ),
        ),
        x * y,
    )

    output = hermeval2d(
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j->ij",
            tensor([4.0, 2.0, 3.0]),
            tensor([4.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_hermeval3d():
    input = rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        tensor([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError):
        hermeval3d(
            a,
            b,
            c[:2],
            einsum(
                "i,j,k->ijk",
                tensor([4.0, 2.0, 3.0]),
                tensor([4.0, 2.0, 3.0]),
                tensor([4.0, 2.0, 3.0]),
            ),
        )

    assert_close(
        hermeval3d(
            a,
            b,
            c,
            einsum(
                "i,j,k->ijk",
                tensor([4.0, 2.0, 3.0]),
                tensor([4.0, 2.0, 3.0]),
                tensor([4.0, 2.0, 3.0]),
            ),
        ),
        x * y * z,
    )

    output = hermeval3d(
        ones([2, 3]),
        ones([2, 3]),
        ones([2, 3]),
        einsum(
            "i,j,k->ijk",
            tensor([4.0, 2.0, 3.0]),
            tensor([4.0, 2.0, 3.0]),
            tensor([4.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_hermevander():
    x = arange(3)
    v = hermevander(
        x,
        3,
    )
    assert v.shape == (3, 4)
    for i in range(4):
        coefficients = tensor([0.0] * i + [1.0])
        assert_close(
            v[..., i],
            hermeval(x, coefficients),
        )

    x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    v = hermevander(
        x,
        3,
    )
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coefficients = tensor([0.0] * i + [1.0])
        assert_close(
            v[..., i],
            hermeval(x, coefficients),
        )


def test_hermevander2d():
    a, b, x3 = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3)

    assert_close(
        dot(
            hermevander2d(
                a,
                b,
                degree=tensor([1, 2]),
            ),
            coefficients.ravel(),
        ),
        hermeval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = hermevander2d(
        [a],
        [b],
        degree=tensor([1, 2]),
    )

    assert output.shape == (1, 5, 6)


def test_hermevander3d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3, 4)

    output = hermevander3d(
        a,
        b,
        c,
        degree=tensor([1, 2, 3]),
    )

    assert_close(
        dot(
            output,
            ravel(coefficients),
        ),
        hermeval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = hermevander3d(
        [a],
        [b],
        [c],
        degree=tensor([1, 2, 3]),
    )

    assert output.shape == (1, 5, 24)


def test_hermeweight():
    assert_close(
        hermeweight(
            linspace(-5, 5, 11),
        ),
        exp(-0.5 * linspace(-5, 5, 11) ** 2),
    )


def test_hermex():
    assert_close(
        hermex,
        tensor([0.0, 1.0]),
    )


def test_hermezero():
    assert_close(
        hermezero,
        tensor([0.0]),
    )


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([3]),
            ),
        ),
        other,
    )

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    assert_close(
        hermfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
        ),
        tensor(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ],
        ).T,
    )

    assert_close(
        hermfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
        ),
        tensor(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight[1::2] = 1.0

    assert_close(
        hermfit(
            input,
            other,
            degree=tensor([3]),
            weight=weight,
        ),
        hermfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        hermfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        hermfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        hermfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
            weight=weight,
        ),
        tensor(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        hermfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        tensor(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        hermfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([1]),
        ),
        tensor([0, 0.5]),
    )

    assert_close(
        hermfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([0, 1]),
        ),
        tensor([0, 0.5]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    assert_close(
        hermfit(
            input,
            other,
            degree=tensor([4]),
        ),
        hermfit(
            input,
            other,
            degree=tensor([0, 2, 4]),
        ),
    )


def test_hermfromroots():
    assert_close(
        hermtrim(
            hermfromroots(
                tensor([]),
            ),
            tol=0.000001,
        ),
        tensor([1.0]),
    )

    for i in range(1, 5):
        roots = cos(linspace(-math.pi, 0, 2 * i + 1)[1::2])
        target = 0

        assert_close(
            herm2poly(
                hermfromroots(
                    roots,
                ),
            )[-1],
            tensor([1.0]),
        )

        assert_close(
            hermval(
                roots,
                hermfromroots(
                    roots,
                ),
            ),
            target,
        )


def test_hermgauss():
    x, w = hermgauss(100)

    v = hermvander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    assert_close(
        vv,
        eye(100),
    )

    assert_close(
        w.sum(),
        tensor(math.sqrt(math.pi)),
    )


def test_hermgrid2d():
    c1d = tensor([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    target = einsum("i,j->ij", y1, y2)
    assert_close(
        hermgrid2d(
            a,
            b,
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

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    assert_close(
        hermgrid3d(
            a,
            b,
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
    with pytest.raises(TypeError):
        hermint(
            tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        hermint(
            tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        hermint(
            tensor([0]),
            order=1,
            k=tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        hermint(
            tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        hermint(
            tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        hermint(
            tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        assert_close(
            hermtrim(
                hermint(
                    tensor([0.0]),
                    order=i,
                    k=([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            tensor([0.0, 0.5]),
        )

    for i in range(5):
        assert_close(
            hermtrim(
                herm2poly(
                    hermint(
                        poly2herm(
                            tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            hermtrim(
                tensor([i] + [0.0] * i + [1.0 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_close(
            hermval(
                tensor([-1.0]),
                hermint(
                    poly2herm(
                        tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        assert_close(
            hermtrim(
                herm2poly(
                    hermint(
                        poly2herm(
                            tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    ),
                ),
                tol=0.000001,
            ),
            hermtrim(
                tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = hermint(
                    target,
                    order=1,
                )

            assert_close(
                hermtrim(
                    hermint(
                        tensor([0.0] * i + [1.0]),
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
            pol = tensor([0.0] * i + [1.0])

            target = pol[:]

            for k in range(j):
                target = hermint(target, order=1, k=[k])

            assert_close(
                hermtrim(
                    hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = hermint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            assert_close(
                hermtrim(
                    hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    # for i in range(5):
    #     for j in range(2, 5):
    #         pol = tensor([0.0] * i + [1.0])
    #         target = pol[:]
    #         for k in range(j):
    #             target = hermint(
    #                 target,
    #                 order=1,
    #                 k=[k],
    #                 scale=2,
    #             )
    #
    #         assert_close(
    #             hermtrim(
    #                 hermint(
    #                     pol,
    #                     order=j,
    #                     k=list(range(j)),
    #                     scale=2,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             hermtrim(
    #                 target,
    #                 tol=0.000001,
    #             ),
    #         )

    c2d = rand(3, 4)

    target = vstack([hermint(c) for c in c2d.T]).T

    assert_close(
        hermint(
            c2d,
            axis=0,
        ),
        target,
    )

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
    assert_close(
        hermline(3, 4),
        tensor([3.0, 2.0]),
    )


def test_hermmul():
    for i in range(5):
        input = linspace(-3, 3, 100)

        val1 = hermval(
            input,
            tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            val2 = hermval(
                input,
                tensor([0.0] * j + [1.0]),
            )

            assert_close(
                hermval(
                    input,
                    hermmul(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                ),
                val1 * val2,
            )


def test_hermmulx():
    assert_close(
        hermtrim(
            hermmulx(
                tensor([0.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0]),
    )

    assert_close(
        hermmulx(
            tensor([1.0]),
        ),
        tensor([0.0, 0.5]),
    )

    for i in range(1, 5):
        assert_close(
            hermmulx(
                tensor([0.0] * i + [1.0]),
            ),
            tensor([0.0] * (i - 1) + [i, 0.0, 0.5]),
        )


def test_hermone():
    assert_close(
        hermone,
        tensor([1.0]),
    )


def test_hermpow():
    for i in range(5):
        for j in range(5):
            assert_close(
                hermtrim(
                    hermpow(
                        arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    functools.reduce(
                        hermmul,
                        [arange(0.0, i + 1)] * j,
                        tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermroots():
    assert_close(
        hermroots(
            tensor([1.0]),
        ),
        tensor([]),
    )

    assert_close(
        hermroots(
            tensor([1.0, 1.0]),
        ),
        tensor([-0.5]),
    )

    for i in range(2, 5):
        input = linspace(-1, 1, i)

        assert_close(
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

            target[i] = target[i] + 1
            target[j] = target[j] - 1

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
    with pytest.raises(ValueError):
        hermtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_close(
        hermtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        hermtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        hermtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        tensor([0.0]),
    )


def test_hermval():
    output = hermval(
        tensor([]),
        tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    input = linspace(-1, 1, 50)

    for coefficient in hermcoefficients:
        ys = [
            *ys,
            polyval(
                input,
                coefficient,
            ),
        ]

    for index in range(10):
        assert_close(
            hermval(
                input,
                tensor([0.0] * index + [1.0]),
            ),
            ys[index],
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = hermval(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = hermval(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = hermval(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_hermval2d():
    c1d = tensor([2.5, 1.0, 0.75])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        hermval2d,
        a,
        b[:2],
        c2d,
    )

    assert_close(
        hermval2d(
            a,
            b,
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
    c1d = tensor([2.5, 1.0, 0.75])
    c3d = einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, hermval3d, a, b, x3[:2], c3d)

    target = y1 * y2 * y3
    assert_close(
        hermval3d(a, b, x3, c3d),
        target,
    )

    z = ones([2, 3])
    assert hermval3d(z, z, z, c3d).shape == (2, 3)


def test_hermvander():
    x = arange(3)

    output = hermvander(
        x,
        degree=3,
    )

    assert output.shape == (3, 4)

    for index in range(4):
        assert_close(
            output[..., index],
            hermval(
                x,
                tensor([0.0] * index + [1.0]),
            ),
        )

    output = hermvander(
        tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=3,
    )

    assert output.shape == (3, 2, 4)

    for index in range(4):
        assert_close(
            output[..., index],
            hermval(
                tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                tensor([0.0] * index + [1.0]),
            ),
        )


def test_hermvander2d():
    a, b, x3 = rand(3, 5) * 2 - 1
    c = rand(2, 3)
    assert_close(
        dot(hermvander2d(a, b, (1, 2)), c.ravel()),
        hermval2d(a, b, c),
    )

    assert hermvander2d([a], [b], (1, 2)).shape == (1, 5, 6)


def test_hermvander3d():
    a, b, x3 = rand(3, 5) * 2 - 1
    c = rand(2, 3, 4)
    assert_close(
        dot(hermvander3d(a, b, x3, (1, 2, 3)), c.ravel()),
        hermval3d(a, b, x3, c),
    )

    assert hermvander3d([a], [b], [x3], (1, 2, 3)).shape == (1, 5, 24)


def test_hermweight():
    assert_close(
        hermweight(linspace(-5, 5, 11)),
        exp(-(linspace(-5, 5, 11) ** 2)),
    )


def test_hermx():
    assert_close(
        hermx,
        tensor([0, 0.5]),
    )


def test_hermzero():
    assert_close(
        hermzero,
        tensor([0.0]),
    )


def test_lag2poly():
    coefficients = [
        tensor([1.0]) / 1,
        tensor([1.0, -1]) / 1,
        tensor([2.0, -4, 1]) / 2,
        tensor([6.0, -18, 9, -1]) / 6,
        tensor([24.0, -96, 72, -16, 1]) / 24,
        tensor([120.0, -600, 600, -200, 25, -1]) / 120,
        tensor([720.0, -4320, 5400, -2400, 450, -36, 1]) / 720,
    ]

    for index in range(7):
        assert_close(
            lag2poly(
                tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_lagadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_close(
                lagtrim(
                    lagadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
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
        lagcompanion(
            tensor([]),
        )

    with pytest.raises(ValueError):
        lagcompanion(
            tensor([1.0]),
        )

    for index in range(1, 5):
        output = lagcompanion(
            tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = lagcompanion(
        tensor([1.0, 2.0]),
    )

    assert output[0, 0] == 1.5


def test_lagder():
    with pytest.raises(TypeError):
        lagder(
            tensor([0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        lagder(
            tensor([0]),
            order=-1,
        )

    for i in range(5):
        assert_close(
            lagtrim(
                lagder(
                    tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            lagtrim(
                tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                lagtrim(
                    lagder(
                        lagint(
                            tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    # for i in range(5):
    #     for j in range(2, 5):
    #         assert_close(
    #             lagtrim(
    #                 lagder(
    #                     lagint(
    #                         tensor([0.0] * i + [1.0]),
    #                         order=j,
    #                         scale=2,
    #                     ),
    #                     order=j,
    #                     scale=0.5,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             lagtrim(
    #                 tensor([0.0] * i + [1.0]),
    #                 tol=0.000001,
    #             ),
    #         )
    #
    # c2d = rand(3, 4)
    #
    # assert_close(
    #     lagder(c2d, axis=0),
    #     vstack([lagder(c) for c in c2d.T]).T,
    # )
    #
    # assert_close(
    #     lagder(
    #         c2d,
    #         axis=1,
    #     ),
    #     vstack([lagder(c) for c in c2d]),
    # )


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = lagdiv(
                lagadd(
                    tensor([0.0] * i + [1.0]),
                    tensor([0.0] * j + [1.0]),
                ),
                tensor([0.0] * i + [1.0]),
            )

            assert_close(
                lagtrim(
                    lagadd(
                        lagmul(
                            quotient,
                            tensor([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    lagadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagdomain():
    assert_close(
        lagdomain,
        tensor([0.0, 1.0]),
    )


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    input = linspace(0, 2, 50)

    other = f(input)

    assert_close(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=tensor([3]),
            ),
        ),
        other,
    )

    assert_close(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    assert_close(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    assert_close(
        lagfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
        ),
        tensor(
            [
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    assert_close(
        lagfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
        ),
        tensor(
            [
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight[1::2] = 1.0

    assert_close(
        lagfit(
            input,
            other,
            degree=tensor([3]),
            weight=weight,
        ),
        lagfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        lagfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        lagfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        lagfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
            weight=weight,
        ),
        tensor(
            [
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ],
        ).T,
    )

    assert_close(
        lagfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        tensor(
            [
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    assert_close(
        lagfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([1]),
        ),
        tensor([1, -1]),
    )

    assert_close(
        lagfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([0, 1]),
        ),
        tensor([1, -1]),
    )


def test_lagfromroots():
    assert_close(
        lagtrim(
            lagfromroots(
                tensor([]),
            ),
            tol=0.000001,
        ),
        tensor([1.0]),
    )

    for i in range(1, 5):
        roots = linspace(-math.pi, 0, 2 * i + 1)

        roots = roots[1::2]

        roots = cos(roots)

        output = lag2poly(
            lagfromroots(
                roots,
            ),
        )

        assert_close(
            output,
            tensor([1.0]),
        )

        output = lagval(
            roots,
            lagfromroots(
                roots,
            ),
        )

        assert_close(
            output,
            tensor([0.0]),
        )


def test_laggauss():
    x, w = laggauss(100)

    v = lagvander(x, 99)
    vv = (v.T * w) @ v
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

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    assert_close(
        laggrid2d(
            a,
            b,
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

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    target = einsum("i,j,k->ijk", y1, y2, y3)
    assert_close(laggrid3d(a, b, x3, c3d), target)

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
    pytest.raises(ValueError, lagint, tensor([0]), lower_bound=[0])
    pytest.raises(ValueError, lagint, tensor([0]), scale=[0])
    pytest.raises(TypeError, lagint, tensor([0]), axis=0.5)

    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        assert_close(
            lagtrim(
                lagint(tensor([0]), order=i, k=k),
                tol=0.000001,
            ),
            [1, -1],
        )

    for i in range(5):
        scale = i + 1
        pol = tensor([0.0] * i + [1.0])
        target = [i] + [0] * i + [1 / scale]
        res = lag2poly(lagint(poly2lag(pol), order=1, k=[i]))
        assert_close(
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
        scale = i + 1
        pol = tensor([0.0] * i + [1.0])
        lagpol = poly2lag(pol)
        assert_close(
            lagval(
                tensor([-1]),
                lagint(
                    lagpol,
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        scale = i + 1
        pol = tensor([0.0] * i + [1.0])
        target = [i] + [0] * i + [2 / scale]
        lagpol = poly2lag(pol)
        assert_close(
            lagtrim(
                lag2poly(lagint(lagpol, order=1, k=[i], scale=2)),
                tol=0.000001,
            ),
            lagtrim(
                target,
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for _ in range(j):
                target = lagint(target, order=1)
            assert_close(
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = lagint(target, order=1, k=[k])
            assert_close(
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = lagint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )
            assert_close(
                lagtrim(
                    lagint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
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
            pol = tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = lagint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )
            assert_close(
                lagtrim(
                    lagint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

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
    assert_close(
        lagline(3.0, 4.0),
        tensor([7.0, -4.0]),
    )


def test_lagmul():
    for i in range(5):
        input = linspace(-3, 3, 100)

        a = lagval(
            input,
            tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = lagval(
                input,
                tensor([0.0] * j + [1.0]),
            )

            assert_close(
                lagval(
                    input,
                    lagtrim(
                        lagmul(
                            tensor([0.0] * i + [1.0]),
                            tensor([0.0] * j + [1.0]),
                        ),
                    ),
                ),
                a * b,
            )


def test_lagmulx():
    assert_close(
        lagtrim(
            lagmulx(
                tensor([0.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0]),
    )

    assert_close(
        lagtrim(
            lagmulx(
                tensor([1.0]),
            ),
            tol=0.000001,
        ),
        tensor([1.0, -1.0]),
    )

    for index in range(1, 5):
        assert_close(
            lagtrim(
                lagmulx(
                    tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            lagtrim(
                tensor(
                    [0.0] * (index - 1) + [-index, 2.0 * index + 1.0, -(index + 1.0)]
                ),
                tol=0.000001,
            ),
        )


def test_lagone():
    assert_close(
        lagone,
        tensor([1.0]),
    )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            assert_close(
                lagtrim(
                    lagpow(
                        arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    functools.reduce(
                        lagmul,
                        [arange(0.0, i + 1)] * j,
                        tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagroots():
    assert_close(
        lagroots(
            tensor([1.0]),
        ),
        tensor([]),
    )

    assert_close(
        lagroots(
            tensor([0.0, 1.0]),
        ),
        tensor([1.0]),
    )

    for index in range(2, 5):
        assert_close(
            lagtrim(
                lagroots(
                    lagfromroots(
                        linspace(0, 3, index),
                    ),
                ),
                tol=0.000001,
            ),
            lagtrim(
                linspace(0, 3, index),
                tol=0.000001,
            ),
        )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            assert_close(
                lagtrim(
                    lagsub(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_lagtrim():
    with pytest.raises(ValueError):
        lagtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_close(
        lagtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        lagtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        lagtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        tensor([0.0]),
    )


def test_lagval():
    output = lagval(
        tensor([]),
        tensor([1.0]),
    )
    assert math.prod(output.shape) == 0

    ys = []

    input = linspace(-1, 1, 50)

    for coefficient in lagcoefficients:
        ys = [
            *ys,
            polyval(
                input,
                coefficient,
            ),
        ]

    for i in range(7):
        assert_close(
            lagval(
                input,
                tensor([0.0] * i + [1.0]),
            ),
            tensor(tensor(ys[i])),
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = lagval(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = lagval(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = lagval(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_lagval2d():
    c1d = tensor([9.0, -14.0, 6.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        lagval2d,
        a,
        b[:2],
        c2d,
    )

    target = y1 * y2
    assert_close(
        lagval2d(
            a,
            b,
            c2d,
        ),
        target,
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

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, lagval3d, a, b, x3[:2], c3d)

    assert_close(
        lagval3d(
            a,
            b,
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
                tensor([0.0] * i + [1.0]),
            ),
        )

    x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    v = lagvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        assert_close(
            v[..., i],
            lagval(
                x,
                tensor([0.0] * i + [1.0]),
            ),
        )


def test_lagvander2d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3)

    assert_close(
        dot(
            lagvander2d(
                a,
                b,
                tensor([1, 2]),
            ),
            ravel(coefficients),
        ),
        lagval2d(a, b, coefficients),
    )

    output = lagvander2d(
        [a],
        [b],
        tensor([1, 2]),
    )

    assert output.shape == (1, 5, 6)


def test_lagvander3d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3, 4)

    assert_close(
        dot(
            lagvander3d(
                a,
                b,
                c,
                degree=tensor([1, 2, 3]),
            ),
            ravel(coefficients),
        ),
        lagval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = lagvander3d(
        [a],
        [b],
        [c],
        degree=tensor([1, 2, 3]),
    )

    assert output.shape == (1, 5, 24)


def test_lagweight():
    assert_close(
        lagweight(linspace(0, 10, 11)),
        exp(-linspace(0, 10, 11)),
    )


def test_lagx():
    assert_close(
        lagx,
        tensor([1.0, -1.0]),
    )


def test_lagzero():
    assert_close(
        lagzero,
        tensor([0.0]),
    )


def test_leg2poly():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 1.0]),
        tensor([-1.0, 0.0, 3.0]) / 2.0,
        tensor([0.0, -3.0, 0.0, 5.0]) / 2.0,
        tensor([3.0, 0.0, -30, 0, 35]) / 8,
        tensor([0.0, 15.0, 0, -70, 0, 63]) / 8,
        tensor([-5.0, 0.0, 105, 0, -315, 0, 231]) / 16,
        tensor([0.0, -35.0, 0, 315, 0, -693, 0, 429]) / 16,
        tensor([35.0, 0.0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        tensor([0.0, 315.0, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    for index in range(10):
        assert_close(
            leg2poly(
                tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_legadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_close(
                legtrim(
                    legadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
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

    for index in range(1, 5):
        output = legcompanion(
            tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    assert legcompanion(tensor([1, 2]))[0, 0] == -0.5


def test_legder():
    with pytest.raises(TypeError):
        legder(
            tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        legder(
            tensor([0.0]),
            order=-1,
        )

    for i in range(5):
        assert_close(
            legtrim(
                legder(
                    tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            legtrim(
                tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                legtrim(
                    legder(
                        legint(
                            tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            assert_close(
                legtrim(
                    legder(
                        legint(
                            tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

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
                    tensor([0.0] * i + [1.0]),
                    tensor([0.0] * j + [1.0]),
                ),
                tensor([0.0] * i + [1.0]),
            )

            assert_close(
                legtrim(
                    legadd(
                        legmul(
                            quotient,
                            tensor([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    legadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legdomain():
    assert_close(
        legdomain,
        tensor([-1.0, 1.0]),
    )


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([3]),
            ),
        ),
        other,
    )

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    assert_close(
        legfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
        ),
        tensor(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        legfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
        ),
        tensor(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight[1::2] = 1.0

    assert_close(
        legfit(
            input,
            other,
            degree=tensor([3]),
            weight=weight,
        ),
        legfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        legfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        legfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        legfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
            weight=weight,
        ),
        tensor(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        legfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        tensor(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    assert_close(
        legfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([1]),
        ),
        tensor([0, 1]),
    )

    assert_close(
        legfit(
            tensor([1, 1j, -1, -1j]),
            tensor([1, 1j, -1, -1j]),
            degree=tensor([0, 1]),
        ),
        tensor([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    assert_close(
        legfit(
            input,
            other,
            degree=tensor([4]),
        ),
        legfit(
            input,
            other,
            degree=tensor([0, 2, 4]),
        ),
    )


def test_legfromroots():
    assert_close(
        legtrim(
            legfromroots(
                tensor([]),
            ),
            tol=0.000001,
        ),
        tensor([1.0]),
    )

    for index in range(1, 5):
        input = linspace(-math.pi, 0, 2 * index + 1)[1::2]

        output = legfromroots(
            cos(
                input,
            ),
        )

        assert output.shape[-1] == index + 1

        assert_close(
            leg2poly(
                legfromroots(
                    cos(
                        input,
                    ),
                )
            )[-1],
            tensor([1.0]),
        )

        assert_close(
            legval(
                cos(
                    input,
                ),
                legfromroots(
                    cos(
                        input,
                    ),
                ),
            ),
            tensor([0.0]),
        )


def test_leggauss():
    x, w = leggauss(100)

    v = legvander(
        x,
        degree=tensor([99]),
    )

    vv = (v.T * w) @ v

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

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    assert_close(
        leggrid2d(
            a,
            b,
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

    x = rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    assert_close(
        leggrid3d(
            a,
            b,
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
    with pytest.raises(TypeError):
        legint(
            tensor([0]),
            0.5,
        )

    with pytest.raises(ValueError):
        legint(
            tensor([0]),
            -1,
        )

    with pytest.raises(ValueError):
        legint(
            tensor([0]),
            1,
            tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        legint(
            tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        legint(
            tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        legint(
            tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        assert_close(
            legtrim(
                legint(
                    tensor([0.0]),
                    order=i,
                    k=tensor([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            tensor([0.0, 1.0]),
        )

    for i in range(5):
        assert_close(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(
                            tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            legtrim(
                tensor([i] + [0.0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        assert_close(
            legval(
                tensor([-1]),
                legint(
                    poly2leg(
                        tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        assert_close(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(
                            tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    )
                ),
                tol=0.000001,
            ),
            legtrim(
                tensor([i] + [0.0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = (tensor([0] * i + [1]))[:]

            for _ in range(j):
                target = legint(
                    target,
                    order=1,
                )

            assert_close(
                legtrim(
                    legint(
                        tensor([0.0] * i + [1.0]),
                        order=j,
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
            target = tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                )

            assert_close(
                legtrim(
                    legint(
                        tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
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
            target = tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            assert_close(
                legtrim(
                    legint(
                        tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
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
            target = tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            assert_close(
                legtrim(
                    legint(
                        tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = rand(3, 4)

    assert_close(
        legint(c2d, axis=0),
        vstack([legint(c) for c in c2d.T]).T,
    )

    target = [legint(c) for c in c2d]

    target = vstack(target)

    assert_close(
        legint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = [legint(c, k=3) for c in c2d]

    target = vstack(target)

    assert_close(
        legint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )

    assert_close(
        legint(
            tensor([1, 2, 3]),
            order=0,
        ),
        tensor([1, 2, 3]),
    )


def test_legline():
    assert_close(
        legline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )

    assert_close(
        legtrim(
            legline(3.0, 0.0),
            tol=0.000001,
        ),
        tensor([3.0]),
    )


def test_legmul():
    for i in range(5):
        input = linspace(-1, 1, 100)

        a = legval(
            input,
            tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = legval(
                input,
                tensor([0.0] * j + [1.0]),
            )

            assert_close(
                legval(
                    input,
                    legmul(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                ),
                a * b,
            )


def test_legmulx():
    assert_close(
        legtrim(
            legmulx(
                tensor([0.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0]),
    )

    assert_close(
        legtrim(
            legmulx(
                tensor([1.0]),
            ),
            tol=0.000001,
        ),
        tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        assert_close(
            legtrim(
                legmulx(
                    tensor([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            tensor([0] * (i - 1) + [i / (2 * i + 1), 0, (i + 1) / (2 * i + 1)]),
        )


def test_legone():
    assert_close(
        legone,
        tensor([1.0]),
    )


def test_legpow():
    for i in range(5):
        for j in range(5):
            assert_close(
                legtrim(
                    legpow(
                        arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    functools.reduce(
                        legmul,
                        [arange(0.0, i + 1)] * j,
                        tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legroots():
    assert_close(
        legroots(
            tensor([1.0]),
        ),
        tensor([]),
    )

    assert_close(
        legroots(tensor([1.0, 2.0])),
        tensor([-0.5]),
    )

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

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            assert_close(
                legtrim(
                    legsub(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
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
        legtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_close(
        legtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        legtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        legtrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        tensor([0.0]),
    )


def test_legval():
    output = legval(
        tensor([]),
        tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in legcoefficients:
        ys = [
            *ys,
            polyval(
                linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for i in range(10):
        assert_close(
            legval(
                linspace(-1, 1, 50),
                tensor([0.0] * i + [1.0]),
            ),
            tensor(ys[i]),
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = legval(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = legval(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = legval(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_legval2d():
    c1d = tensor([2.0, 2.0, 2.0])
    c2d = einsum("i,j->ij", c1d, c1d)

    x = rand(3, 5) * 2 - 1
    y = polyval(x, tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        legval2d,
        a,
        b[:2],
        c2d,
    )

    assert_close(
        legval2d(
            a,
            b,
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

    x = rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(x, tensor([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, legval3d, a, b, x3[:2], c3d)

    assert_close(
        legval3d(
            a,
            b,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    z = ones([2, 3])

    assert legval3d(z, z, z, c3d).shape == (2, 3)


def test_legvander():
    x = arange(3)

    v = legvander(
        x,
        degree=3,
    )

    assert v.shape == (3, 4)

    for index in range(4):
        assert_close(
            v[..., index],
            legval(
                x,
                tensor([0.0] * index + [1.0]),
            ),
        )

    x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    v = legvander(
        x,
        degree=3,
    )

    assert v.shape == (3, 2, 4)

    for index in range(4):
        assert_close(
            v[..., index],
            legval(
                x,
                tensor([0.0] * index + [1.0]),
            ),
        )

    with pytest.raises(ValueError):
        legvander(
            tensor([1, 2, 3]),
            -1,
        )


def test_legvander2d():
    a, b, x3 = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3)

    assert_close(
        dot(
            legvander2d(
                a,
                b,
                degree=tensor([1, 2]),
            ),
            ravel(coefficients),
        ),
        legval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = legvander2d(
        [a],
        [b],
        degree=tensor([1, 2]),
    )

    assert output.shape == (1, 5, 6)


def test_legvander3d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3, 4)

    target = legval3d(
        a,
        b,
        c,
        coefficients,
    )

    assert_close(
        dot(
            legvander3d(
                a,
                b,
                c,
                degree=tensor([1, 2, 3]),
            ),
            ravel(coefficients),
        ),
        target,
    )

    output = legvander3d(
        [a],
        [b],
        [c],
        degree=tensor([1, 2, 3]),
    )

    assert output.shape == (1, 5, 24)


def test_legweight():
    assert_close(
        legweight(
            linspace(-1, 1, 11),
        ),
        tensor([1.0]),
    )


def test_legx():
    assert_close(
        legx,
        tensor([0.0, 1.0]),
    )


def test_legzero():
    assert_close(
        legzero,
        tensor([0.0]),
    )


def test_poly2cheb():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 1]),
        tensor([-1.0, 0, 2]),
        tensor([0.0, -3, 0, 4]),
        tensor([1.0, 0, -8, 0, 8]),
        tensor([0.0, 5, 0, -20, 0, 16]),
        tensor([-1.0, 0, 18, 0, -48, 0, 32]),
        tensor([0.0, -7, 0, 56, 0, -112, 0, 64]),
        tensor([1.0, 0, -32, 0, 160, 0, -256, 0, 128]),
        tensor([0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
    ]

    for index in range(10):
        assert_close(
            poly2cheb(
                coefficients[index],
            ),
            tensor([0.0] * index + [1.0]),
        )


def test_poly2herm():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 2]),
        tensor([-2.0, 0, 4]),
        tensor([0.0, -12, 0, 8]),
        tensor([12.0, 0, -48, 0, 16]),
        tensor([0.0, 120, 0, -160, 0, 32]),
        tensor([-120.0, 0, 720, 0, -480, 0, 64]),
        tensor([0.0, -1680, 0, 3360, 0, -1344, 0, 128]),
        tensor([1680.0, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        tensor([0.0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    for index in range(10):
        assert_close(
            hermtrim(
                poly2herm(
                    coefficients[index],
                ),
                tol=0.000001,
            ),
            tensor([0.0] * index + [1.0]),
        )


def test_poly2herme():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 1]),
        tensor([-1.0, 0, 1]),
        tensor([0.0, -3, 0, 1]),
        tensor([3.0, 0, -6, 0, 1]),
        tensor([0.0, 15, 0, -10, 0, 1]),
        tensor([-15.0, 0, 45, 0, -15, 0, 1]),
        tensor([0.0, -105, 0, 105, 0, -21, 0, 1]),
        tensor([105.0, 0, -420, 0, 210, 0, -28, 0, 1]),
        tensor([0.0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    for index in range(10):
        assert_close(
            poly2herme(
                coefficients[index],
            ),
            tensor([0.0] * index + [1.0]),
        )


def test_poly2lag():
    coefficients = [
        tensor([1.0]) / 1.0,
        tensor([1.0, -1.0]) / 1.0,
        tensor([2.0, -4.0, 1.0]) / 2.0,
        tensor([6.0, -18.0, 9.0, -1.0]) / 6.0,
        tensor([24.0, -96.0, 72.0, -16.0, 1.0]) / 24.0,
        tensor([120.0, -600.0, 600.0, -200.0, 25.0, -1.0]) / 120.0,
        tensor([720.0, -4320.0, 5400.0, -2400.0, 450.0, -36.0, 1.0]) / 720.0,
    ]

    for index in range(7):
        assert_close(
            poly2lag(
                coefficients[index],
            ),
            tensor([0.0] * index + [1.0]),
        )


def test_poly2leg():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 1]),
        tensor([-1.0, 0, 3]) / 2,
        tensor([0.0, -3, 0, 5]) / 2,
        tensor([3.0, 0, -30, 0, 35]) / 8,
        tensor([0.0, 15, 0, -70, 0, 63]) / 8,
        tensor([-5.0, 0, 105, 0, -315, 0, 231]) / 16,
        tensor([0.0, -35, 0, 315, 0, -693, 0, 429]) / 16,
        tensor([35.0, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        tensor([0.0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    for index in range(10):
        assert_close(
            poly2leg(
                coefficients[index],
            ),
            tensor([0.0] * index + [1.0]),
        )


def test_polyadd():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_close(
                polytrim(
                    polyadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
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
            tensor([0.0] * i + [1.0]),
        )

        assert output.shape == (i, i)

    output = polycompanion(
        tensor([1, 2]),
    )

    assert output[0, 0] == -0.5


def test_polydiv():
    quotient, remainder = polydiv(
        tensor([2.0]),
        tensor([2.0]),
    )

    assert_close(
        quotient,
        tensor([1.0]),
    )

    assert_close(
        remainder,
        tensor([0.0]),
    )

    quotient, remainder = polydiv(
        tensor([2.0, 2.0]),
        tensor([2.0]),
    )

    assert_close(
        quotient,
        tensor([1.0, 1.0]),
    )

    assert_close(
        remainder,
        tensor([0.0]),
    )

    for j in range(5):
        for k in range(5):
            input = tensor([0.0] * j + [1.0, 2.0])
            other = tensor([0.0] * k + [1.0, 2.0])

            quotient, remainder = polydiv(
                polyadd(
                    input,
                    other,
                ),
                input,
            )

            assert_close(
                polytrim(
                    polyadd(
                        polymul(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    polyadd(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )


def test_polydomain():
    assert_close(
        polydomain,
        tensor([-1.0, 1.0]),
    )


def test_polyfit():
    def f(x: Tensor) -> Tensor:
        return x * (x - 1) * (x - 2)

    def g(x: Tensor) -> Tensor:
        return x**4 + x**2 + 1

    input = linspace(0, 2, 50)

    other = f(input)

    assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=tensor([3]),
            ),
        ),
        other,
    )

    assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    assert_close(
        polyfit(
            input,
            tensor([other, other]).T,
            degree=tensor([3]),
        ),
        tensor(
            [
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    assert_close(
        polyfit(
            input,
            tensor([other, other]).T,
            degree=tensor([0, 1, 2, 3]),
        ),
        tensor(
            [
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    weight = zeros_like(input)

    weight[1::2] = 1.0

    assert_close(
        polyfit(
            input,
            other.at[0::2].set(0),
            degree=3,
            weight=weight,
        ),
        polyfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        polyfit(
            input,
            other.at[0::2].set(0),
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        polyfit(
            input,
            other,
            degree=tensor([0, 1, 2, 3]),
        ),
    )

    assert_close(
        polyfit(
            input,
            tensor([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
            degree=3,
            weight=weight,
        ),
        tensor(
            [
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    assert_close(
        polyfit(
            input,
            tensor([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
            degree=tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        tensor(
            [
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
                polyfit(
                    input,
                    other,
                    degree=tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
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
            tensor([1, 1j, -1, -0 - 1j]),
            tensor([1, 1j, -1, -0 - 1j]),
            (0, 1),
        ),
        tensor([0, 1]),
    )

    input = linspace(-1, 1, 50)

    other = g(input)

    assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=tensor([4]),
            ),
        ),
        other,
    )

    assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    assert_close(
        polyfit(
            input,
            other,
            degree=tensor([4]),
        ),
        polyfit(
            input,
            other,
            degree=tensor([0, 2, 4]),
        ),
    )


def test_polyfromroots():
    coefficients = [
        tensor([1.0]),
        tensor([0.0, 1]),
        tensor([-1.0, 0, 2]),
        tensor([0.0, -3, 0, 4]),
        tensor([1.0, 0, -8, 0, 8]),
        tensor([0.0, 5, 0, -20, 0, 16]),
        tensor([-1.0, 0, 18, 0, -48, 0, 32]),
        tensor([0.0, -7, 0, 56, 0, -112, 0, 64]),
        tensor([1.0, 0, -32, 0, 160, 0, -256, 0, 128]),
        tensor([0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
    ]

    assert_close(
        polytrim(
            polyfromroots(
                tensor([]),
            ),
            tol=0.000001,
        ),
        tensor([1.0]),
    )

    for index in range(1, 5):
        input = linspace(-math.pi, 0.0, 2 * index + 1)

        input = input[1::2]

        input = cos(input)

        output = polyfromroots(input) * 2 ** (index - 1)

        assert_close(
            polytrim(
                output,
                tol=0.000001,
            ),
            polytrim(
                coefficients[index],
                tol=0.000001,
            ),
        )


def test_polygrid2d():
    x = rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        polygrid2d(
            a,
            b,
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
    x = rand(3, 5) * 2 - 1

    y = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    a, b, x3 = x
    y1, y2, y3 = y

    assert_close(
        polygrid3d(
            a,
            b,
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


def test_polyline():
    assert_close(
        polyline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )

    assert_close(
        polyline(3.0, 0.0),
        tensor([3.0, 0.0]),
    )


def test_polymul():
    for j in range(5):
        for k in range(5):
            target = zeros(j + k + 1)

            target[j + k] = target[j + k] + 1

            assert_close(
                polytrim(
                    polymul(
                        tensor([0.0] * j + [1.0]),
                        tensor([0.0] * k + [1.0]),
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
            tensor([0.0]),
        ),
        tensor([0.0, 0.0]),
    )

    assert_close(
        polymulx(
            tensor([1.0]),
        ),
        tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        assert_close(
            polymulx(
                tensor([0.0] * i + [1.0]),
            ),
            tensor([0.0] * (i + 1) + [1.0]),
        )


def test_polyone():
    assert_close(
        polyone,
        tensor([1.0]),
    )


def test_polypow():
    for i in range(5):
        for j in range(5):
            assert_close(
                polytrim(
                    polypow(
                        arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    functools.reduce(
                        polymul,
                        [arange(0.0, i + 1)] * j,
                        tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_polyroots():
    assert_close(
        polyroots(tensor([1.0])),
        tensor([]),
    )

    assert_close(
        polyroots(tensor([1.0, 2.0])),
        tensor([-0.5]),
    )

    for index in range(2, 5):
        input = linspace(-1, 1, index)

        assert_close(
            polytrim(
                polyroots(
                    polyfromroots(
                        input,
                    ),
                ),
                tol=0.000001,
            ),
            polytrim(
                input,
                tol=0.000001,
            ),
        )


def test_polysub():
    for i in range(5):
        for j in range(5):
            target = zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            assert_close(
                polytrim(
                    polysub(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polytrim():
    with pytest.raises(ValueError):
        polytrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_close(
        polytrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_close(
        polytrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_close(
        polytrim(
            tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        tensor([0.0]),
    )


def test_polyval():
    output = polyval(
        tensor([]),
        tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    y = []

    input = linspace(-1, 1, 50)

    for index in range(5):
        y = [
            *y,
            input**index,
        ]

    for index in range(5):
        assert_close(
            polyval(
                input,
                tensor([0.0] * index + [1.0]),
            ),
            y[index],
        )

    assert_close(
        polyval(
            input,
            tensor([0, -1, 0, 1]),
        ),
        input * (input**2 - 1),
    )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = polyval(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_polyval2d():
    x = rand(3, 5) * 2 - 1

    a, b, c = x

    y1, y2, y3 = polyval(
        x,
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        polyval2d(
            a,
            b,
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
    input = rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        tensor([1.0, 2.0, 3.0]),
    )

    assert_close(
        polyval3d(
            a,
            b,
            c,
            einsum(
                "i,j,k->ijk",
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
                tensor([1.0, 2.0, 3.0]),
            ),
        ),
        x * y * z,
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
    with pytest.raises(ValueError):
        polyvalfromroots(
            tensor([1.0]),
            tensor([1.0]),
            tensor=False,
        )

    output = polyvalfromroots(
        tensor([]),
        tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (0,)

    output = polyvalfromroots(
        tensor([]),
        tensor([[1.0] * 5]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (5, 0)

    assert_close(
        polyvalfromroots(
            tensor([1.0]),
            tensor([1.0]),
        ),
        tensor([0.0]),
    )

    output = polyvalfromroots(
        tensor([1.0]),
        ones([3, 3]),
    )

    assert output.shape == (3, 1)

    input = linspace(-1, 1, 50)

    evaluations = []

    for i in range(5):
        evaluations = [*evaluations, input**i]

    for i in range(1, 5):
        target = evaluations[i]

        assert_close(
            polyvalfromroots(
                input,
                tensor([0.0] * i),
            ),
            target,
        )

    assert_close(
        polyvalfromroots(
            input,
            tensor([-1.0, 0.0, 1.0]),
        ),
        input * (input - 1.0) * (input + 1.0),
    )

    for i in range(3):
        shape = (2,) * i

        input = zeros(shape)

        output = polyvalfromroots(
            input,
            tensor([1.0]),
        )

        assert output.shape == shape

        output = polyvalfromroots(
            input,
            tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = polyvalfromroots(
            input,
            tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

    ptest = tensor([15.0, 2.0, -16.0, -2.0, 1.0])

    r = polyroots(ptest)

    assert_close(
        polyval(
            input,
            ptest,
        ),
        polyvalfromroots(
            input,
            r,
        ),
    )

    x = arange(-3, 2)

    r = randint(-5, 5, (3, 5)).to(float64)

    target = empty(r.shape[1:])

    for j in range(math.prod(target.shape)):
        target[j] = polyvalfromroots(
            x[j],
            r[:, j],
        )

    assert_close(
        polyvalfromroots(x, r, tensor=False),
        target,
    )

    x = vstack([x, 2 * x])

    target = empty(r.shape[1:] + x.shape)

    for j in range(r.shape[1]):
        for k in range(x.shape[0]):
            target[j, k, :] = polyvalfromroots(x[k], r[:, j])

    assert_close(
        polyvalfromroots(x, r, tensor=True),
        target,
    )


def test_polyvander():
    output = polyvander(
        arange(3.0),
        degree=tensor([3]),
    )

    assert output.shape == (3, 4)

    for i in range(4):
        assert_close(
            output[..., i],
            polyval(
                arange(3),
                tensor([0.0] * i + [1.0]),
            ),
        )

    output = polyvander(
        tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=tensor([3]),
    )

    assert output.shape == (3, 2, 4)

    for i in range(4):
        assert_close(
            output[..., i],
            polyval(
                tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                tensor([0.0] * i + [1.0]),
            ),
        )

    with pytest.raises(ValueError):
        polyvander(
            arange(3),
            degree=tensor([-1]),
        )


def test_polyvander2d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3)

    assert_close(
        dot(
            polyvander2d(
                a,
                b,
                degree=tensor([1, 2]),
            ),
            ravel(coefficients),
        ),
        polyval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = polyvander2d(
        tensor([a]),
        tensor([b]),
        degree=tensor([1, 2]),
    )

    assert output.shape == (1, 5, 6)


def test_polyvander3d():
    a, b, c = rand(3, 5) * 2 - 1

    coefficients = rand(2, 3, 4)

    assert_close(
        dot(
            polyvander3d(
                a,
                b,
                c,
                degree=tensor([1.0, 2.0, 3.0]),
            ),
            ravel(coefficients),
        ),
        polyval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = polyvander3d(
        tensor([a]),
        tensor([b]),
        tensor([c]),
        degree=tensor([1.0, 2.0, 3.0]),
    )

    assert output.shape == (1, 5, 24)


def test_polyx():
    assert_close(
        polyx,
        tensor([0.0, 1.0]),
    )


def test_polyzero():
    assert_close(
        polyzero,
        tensor([0.0]),
    )
