import functools
import math

import pytest
import torch
import torch.testing
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
    chebadd,
    chebcompanion,
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
    legdiv,
    legdomain,
    legfit,
    legfromroots,
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
from torch import Tensor

torch.set_default_dtype(torch.float64)


def test__c_series_to_z_series():
    for index in range(5):
        torch.testing.assert_close(
            _c_series_to_z_series(
                torch.tensor([2.0] + [1.0] * index),
            ),
            torch.tensor([0.5] * index + [2.0] + [0.5] * index),
        )


def test__fit():
    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([-1.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([[1.0]]),
            torch.tensor([1.0]),
            degree=torch.tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([]),
            torch.tensor([1.0]),
            degree=torch.tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([[[1.0]]]),
            degree=torch.tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0, 2.0]),
            degree=torch.tensor([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([0.0]),
            weight=[[1.0]],
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([0.0]),
            weight=torch.tensor([1.0, 1.0]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([-1.0]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([2.0, -1.0, 6.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            degree=torch.tensor([]),
        )


def test__get_domain():
    torch.testing.assert_close(
        _get_domain(
            torch.tensor([1.0, 10.0, 3.0, -1.0]),
        ),
        torch.tensor([-1.0, 10.0]),
    )

    torch.testing.assert_close(
        _get_domain(
            torch.tensor([1 + 1j, 1 - 1j, 0.0, 2.0]),
        ),
        torch.tensor([-0 - 1j, 2 + 1j]),
    )


def test__map_domain():
    torch.testing.assert_close(
        _map_domain(
            torch.tensor([0.0, 4.0]),
            torch.tensor([0.0, 4.0]),
            torch.tensor([1.0, 3.0]),
        ),
        torch.tensor([1.0, 3.0]),
    )

    torch.testing.assert_close(
        _map_domain(
            torch.tensor([-0 - 1j, 2 + 1j]),
            torch.tensor([-0 - 1j, 2 + 1j]),
            torch.tensor([-2.0, 2.0]),
        ),
        torch.tensor([-2 + 0j, 2 + 0j]),
    )

    torch.testing.assert_close(
        _map_domain(
            torch.tensor([[0.0, 4.0], [0.0, 4.0]]),
            torch.tensor([0.0, 4.0]),
            torch.tensor([1.0, 3.0]),
        ),
        torch.tensor([[1.0, 3.0], [1.0, 3.0]]),
    )


def test__map_parameters():
    torch.testing.assert_close(
        _map_parameters(
            torch.tensor([0.0, 4.0]),
            torch.tensor([1.0, 3.0]),
        ),
        torch.tensor([1.0, 0.5]),
    )

    torch.testing.assert_close(
        _map_parameters(
            torch.tensor([-1j, 2 + 1j]),
            torch.tensor([-2 + 0j, 2 + 0j]),
        ),
        torch.tensor([-1 + 1j, 1 - 1j]),
    )


def test__pow():
    with pytest.raises(ValueError):
        _pow(
            (),
            torch.tensor([1.0, 2.0, 3.0]),
            exponent=5,
            maximum_exponent=4,
        )


def test__trim_coefficients():
    with pytest.raises(ValueError):
        _trim_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        _trim_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        _trim_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        _trim_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )


def test__trim_sequence():
    for _ in range(5):
        torch.testing.assert_close(
            _trim_sequence(
                torch.tensor([1.0] + [0.0] * 5),
            ),
            torch.tensor([1.0]),
        )


def test__vandermonde():
    with pytest.raises(ValueError):
        _vandermonde(
            (),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([90.0]),
        )

    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (),
            torch.tensor([90.65]),
        )

    with pytest.raises(ValueError):
        _vandermonde(
            (),
            (),
            torch.tensor([]),
        )


def test__z_series_to_c_series():
    for index in range(5):
        torch.testing.assert_close(
            _z_series_to_c_series(
                torch.tensor([0.5] * index + [2.0] + [0.5] * index),
            ),
            torch.tensor([2.0] + [1.0] * index),
        )


def test_chebadd():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            torch.testing.assert_close(
                chebtrim(
                    chebadd(
                        torch.tensor([0.0] * j + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
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
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        chebcompanion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = chebcompanion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = chebcompanion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.5


def test_chebdiv():
    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            quotient, remainder = chebdiv(
                chebadd(
                    input,
                    other,
                ),
                input,
            )

            torch.testing.assert_close(
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
    torch.testing.assert_close(
        chebdomain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )


def test_chebfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        chebval(
            input,
            chebfit(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    # assert_close(
    #     chebfit(
    #         input,
    #         stack([other, other]).T,
    #         degree=4,
    #     ),
    #     stack(
    #         [
    #             chebfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #             chebfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )

    torch.testing.assert_close(
        chebfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    # weight = zeros_like(input)
    #
    # weight[1::2] = 1.0
    #
    # assert_close(
    #     chebfit(
    #         input,
    #         other,
    #         degree=tensor([3]),
    #         weight=weight,
    #     ),
    #     chebfit(
    #         input,
    #         other,
    #         degree=tensor([0, 1, 2, 3]),
    #     ),
    # )
    #
    # assert_close(
    #     chebfit(
    #         input,
    #         other,
    #         degree=tensor([0, 1, 2, 3]),
    #         weight=weight,
    #     ),
    #     chebfit(
    #         input,
    #         other,
    #         degree=tensor([0, 1, 2, 3]),
    #     ),
    # )
    #
    # assert_close(
    #     chebfit(
    #         input,
    #         tensor([other, other]).T,
    #         degree=tensor([3]),
    #         weight=weight,
    #     ),
    #     tensor(
    #         [
    #             chebfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #             chebfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )
    #
    # assert_close(
    #     chebfit(
    #         input,
    #         tensor([other, other]).T,
    #         degree=tensor([0, 1, 2, 3]),
    #         weight=weight,
    #     ),
    #     tensor(
    #         [
    #             chebfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #             chebfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #         ],
    #     ).T,
    # )
    #
    # assert_close(
    #     chebfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([1]),
    #     ),
    #     tensor([0, 1]),
    # )
    #
    # assert_close(
    #     chebfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([0, 1]),
    #     ),
    #     tensor([0, 1]),
    # )
    #
    # input = linspace(-1, 1, 50)
    #
    # other = g(input)
    #
    # assert_close(
    #     chebval(
    #         input,
    #         chebfit(
    #             input,
    #             other,
    #             degree=tensor([4]),
    #         ),
    #     ),
    #     other,
    # )
    #
    # assert_close(
    #     chebval(
    #         input,
    #         chebfit(
    #             input,
    #             other,
    #             degree=tensor([0, 2, 4]),
    #         ),
    #     ),
    #     other,
    # )
    #
    # assert_close(
    #     chebfit(
    #         input,
    #         other,
    #         degree=tensor([4]),
    #     ),
    #     chebfit(
    #         input,
    #         other,
    #         degree=tensor([0, 2, 4]),
    #     ),
    # )


def test_chebfromroots():
    torch.testing.assert_close(
        chebtrim(
            chebfromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for index in range(1, 5):
        input = chebfromroots(
            torch.cos(torch.linspace(-math.pi, 0.0, 2 * index + 1)[1::2]),
        )

        input = input * 2 ** (index - 1)

        torch.testing.assert_close(
            chebtrim(
                input,
                tol=0.000001,
            ),
            chebtrim(
                torch.tensor([0.0] * index + [1.0]),
                tol=0.000001,
            ),
        )


def test_chebgauss():
    output, weight = chebgauss(100)

    vandermonde = chebvander(
        output,
        degree=torch.tensor([99]),
    )

    u = (vandermonde.T * weight) @ vandermonde

    v = 1 / torch.sqrt(u.diagonal())

    torch.testing.assert_close(
        v[:, None] * u * v,
        torch.eye(100),
    )

    torch.testing.assert_close(
        weight.sum(),
        torch.tensor(math.pi),
    )


def test_chebgrid2d():
    x = torch.rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        chebgrid2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
            ),
        ),
        torch.einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    res = chebgrid2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert res.shape == (2, 3) * 2


def test_chebgrid3d():
    x = torch.rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        chebgrid3d(
            a,
            b,
            x3,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
            ),
        ),
        torch.einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    output = chebgrid3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_chebint():
    with pytest.raises(TypeError):
        chebint(
            torch.tensor([0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        chebint(
            torch.tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        chebint(
            torch.tensor([0.0]),
            order=1,
            k=torch.tensor([0.0, 0.0]),
        )

    with pytest.raises(TypeError):
        chebint(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            chebtrim(
                chebint(
                    torch.tensor([0.0]),
                    order=i,
                    k=torch.tensor([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0, 1.0]),
        )

    # for i in range(5):
    #     assert_close(
    #         chebtrim(
    #             cheb2poly(
    #                 chebint(
    #                     poly2cheb(
    #                         tensor([0.0] * i + [1.0]),
    #                     ),
    #                     order=1,
    #                     k=[i],
    #                 ),
    #             ),
    #             tol=0.000001,
    #         ),
    #         chebtrim(
    #             tensor([i] + [0] * i + [1 / (i + 1)]),
    #             tol=0.000001,
    #         ),
    #     )
    #
    # for i in range(5):
    #     assert_close(
    #         chebval(
    #             tensor([-1]),
    #             chebint(
    #                 poly2cheb(
    #                     tensor([0.0] * i + [1.0]),
    #                 ),
    #                 order=1,
    #                 k=[i],
    #                 lower_bound=-1,
    #             ),
    #         ),
    #         i,
    #     )
    #
    # for i in range(5):
    #     assert_close(
    #         chebtrim(
    #             cheb2poly(
    #                 chebint(
    #                     poly2cheb(
    #                         tensor([0.0] * i + [1.0]),
    #                     ),
    #                     order=1,
    #                     k=[i],
    #                     scale=2,
    #                 )
    #             ),
    #             tol=0.000001,
    #         ),
    #         chebtrim(
    #             tensor([i] + [0] * i + [2 / (i + 1)]),
    #             tol=0.000001,
    #         ),
    #     )
    #
    # for i in range(5):
    #     for j in range(2, 5):
    #         input = tensor([0.0] * i + [1.0])
    #         target = input[:]
    #
    #         for _ in range(j):
    #             target = chebint(
    #                 target,
    #                 order=1,
    #             )
    #
    #         assert_close(
    #             chebtrim(
    #                 chebint(
    #                     input,
    #                     order=j,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             chebtrim(
    #                 target,
    #                 tol=0.000001,
    #             ),
    #         )
    #
    # for i in range(5):
    #     for j in range(2, 5):
    #         input = tensor([0.0] * i + [1.0])
    #
    #         target = input[:]
    #
    #         for k in range(j):
    #             target = chebint(
    #                 target,
    #                 order=1,
    #                 k=[k],
    #             )
    #
    #         assert_close(
    #             chebtrim(
    #                 chebint(
    #                     input,
    #                     order=j,
    #                     k=list(range(j)),
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             chebtrim(
    #                 target,
    #                 tol=0.000001,
    #             ),
    #         )
    #
    # for i in range(5):
    #     for j in range(2, 5):
    #         input = tensor([0.0] * i + [1.0])
    #
    #         target = input[:]
    #
    #         for k in range(j):
    #             target = chebint(
    #                 target,
    #                 order=1,
    #                 k=[k],
    #                 lower_bound=-1,
    #             )
    #
    #         assert_close(
    #             chebtrim(
    #                 chebint(
    #                     input,
    #                     order=j,
    #                     k=list(range(j)),
    #                     lower_bound=-1,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             chebtrim(
    #                 target,
    #                 tol=0.000001,
    #             ),
    #         )
    #
    # for i in range(5):
    #     for j in range(2, 5):
    #         input = tensor([0.0] * i + [1.0])
    #
    #         target = input[:]
    #
    #         for k in range(j):
    #             target = chebint(
    #                 target,
    #                 order=1,
    #                 k=[k],
    #                 scale=2,
    #             )
    #
    #         assert_close(
    #             chebtrim(
    #                 chebint(
    #                     input,
    #                     order=j,
    #                     k=list(range(j)),
    #                     scale=2,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             chebtrim(
    #                 target,
    #                 tol=0.000001,
    #             ),
    #         )
    #
    # c2d = rand(3, 4)
    #
    # assert_close(
    #     chebint(
    #         c2d,
    #         axis=0,
    #     ),
    #     vstack([chebint(c) for c in c2d.T]).T,
    # )
    #
    # assert_close(
    #     chebint(
    #         c2d,
    #         axis=1,
    #     ),
    #     vstack([chebint(c) for c in c2d]),
    # )
    #
    # assert_close(
    #     chebint(
    #         c2d,
    #         k=3,
    #         axis=1,
    #     ),
    #     vstack([chebint(c, k=3) for c in c2d]),
    # )


def test_chebinterpolate():
    def f(x):
        return x * (x - 1) * (x - 2)

    with pytest.raises(ValueError):
        chebinterpolate(f, -1)

    for i in range(1, 5):
        assert chebinterpolate(f, i).shape == (i + 1,)

    def powx(x, p):
        return x**p

    x = torch.linspace(-1, 1, 10)

    for i in range(0, 10):
        for j in range(0, i + 1):
            c = chebinterpolate(
                powx,
                i,
                (j,),
            )

            torch.testing.assert_close(
                chebval(x, c),
                powx(x, j),
            )


def test_chebline():
    torch.testing.assert_close(
        chebline(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )


def test_chebmul():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(j + k + 1)

            target[abs(j + k)] = target[abs(j + k)] + 0.5
            target[abs(j - k)] = target[abs(j - k)] + 0.5

            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            torch.testing.assert_close(
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
    torch.testing.assert_close(
        chebtrim(
            chebmulx(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        chebtrim(
            chebmulx(
                torch.tensor([1.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0, 1.0]),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            chebtrim(
                chebmulx(
                    torch.tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0] * (index - 1) + [0.5, 0, 0.5]),
        )


def test_chebone():
    torch.testing.assert_close(
        chebone,
        torch.tensor([1.0]),
        check_dtype=False,
    )


def test_chebpow():
    for j in range(5):
        for k in range(5):
            torch.testing.assert_close(
                chebtrim(
                    chebpow(
                        torch.arange(0.0, j + 1),
                        k,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    functools.reduce(
                        chebmul,
                        [torch.arange(0.0, j + 1)] * k,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebpts1():
    with pytest.raises(ValueError):
        chebpts1(0)

    torch.testing.assert_close(
        chebpts1(1),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        chebpts1(2),
        torch.tensor([-0.70710678118654746, 0.70710678118654746]),
    )

    torch.testing.assert_close(
        chebpts1(3),
        torch.tensor([-0.86602540378443871, 0, 0.86602540378443871]),
    )

    torch.testing.assert_close(
        chebpts1(4),
        torch.tensor([-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]),
    )


def test_chebpts2():
    with pytest.raises(ValueError):
        chebpts2(1.5)

    with pytest.raises(ValueError):
        chebpts2(1)

    torch.testing.assert_close(
        chebpts2(2),
        torch.tensor([-1.0, 1.0]),
    )

    torch.testing.assert_close(
        chebpts2(3),
        torch.tensor([-1.0, 0.0, 1.0]),
    )

    torch.testing.assert_close(
        chebpts2(4),
        torch.tensor([-1.0, -0.5, 0.5, 1.0]),
    )

    torch.testing.assert_close(
        chebpts2(5),
        torch.tensor([-1.0, -0.707106781187, 0, 0.707106781187, 1.0]),
    )


def test_chebroots():
    torch.testing.assert_close(
        chebroots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        chebroots(
            torch.tensor([1.0, 2.0]),
        ),
        torch.tensor([-0.5]),
    )

    for i in range(2, 5):
        torch.testing.assert_close(
            chebtrim(
                chebroots(
                    chebfromroots(
                        torch.linspace(-1, 1, i),
                    )
                ),
                tol=0.000001,
            ),
            chebtrim(
                torch.linspace(-1, 1, i),
                tol=0.000001,
            ),
        )


def test_chebsub():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] - 1

            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            torch.testing.assert_close(
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
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        chebtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        chebtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        chebtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )


def test_chebval():
    chebcoefficients = [
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([-1, 0, 2]),
        torch.tensor([0, -3, 0, 4]),
        torch.tensor([1, 0, -8, 0, 8]),
        torch.tensor([0, 5, 0, -20, 0, 16]),
        torch.tensor([-1, 0, 18, 0, -48, 0, 32]),
        torch.tensor([0, -7, 0, 56, 0, -112, 0, 64]),
        torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
        torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
    ]

    output = chebval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in chebcoefficients:
        ys = [
            *ys,
            polyval(
                torch.linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for index in range(10):
        torch.testing.assert_close(
            chebval(
                torch.linspace(-1, 1, 50),
                torch.tensor([0.0] * index + [1.0]),
            ),
            torch.tensor(ys[index]),
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = chebval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = chebval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = chebval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_chebval2d():
    x = torch.rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    pytest.raises(
        ValueError,
        chebval2d,
        a,
        b[:2],
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
        ),
    )

    torch.testing.assert_close(
        chebval2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
            ),
        ),
        y1 * y2,
    )

    res = chebval2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert res.shape == (2, 3)


def test_chebval3d():
    c3d = torch.einsum(
        "i,j,k->ijk",
        torch.tensor([2.5, 2.0, 1.5]),
        torch.tensor([2.5, 2.0, 1.5]),
        torch.tensor([2.5, 2.0, 1.5]),
    )

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    pytest.raises(ValueError, chebval3d, a, b, x3[:2], c3d)

    torch.testing.assert_close(
        chebval3d(
            a,
            b,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    output = chebval3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        c3d,
    )

    assert output.shape == (2, 3)


def test_chebvander():
    v = chebvander(
        torch.arange(3),
        degree=torch.tensor([3]),
    )

    assert v.shape == (3, 4)

    for i in range(4):
        torch.testing.assert_close(
            v[..., i],
            chebval(
                torch.arange(3),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    v = chebvander(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=torch.tensor([3]),
    )

    assert v.shape == (3, 2, 4)

    for i in range(4):
        torch.testing.assert_close(
            v[..., i],
            chebval(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )


def test_chebvander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = chebvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        chebval2d(
            a,
            b,
            coefficients,
        ),
    )

    van = chebvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert van.shape == (5, 6)


def test_chebvander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = chebvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        chebval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = chebvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)


def test_chebweight():
    x = torch.linspace(-1, 1, 11)[1:-1]

    torch.testing.assert_close(
        chebweight(x),
        1.0 / (torch.sqrt(1 + x) * torch.sqrt(1 - x)),
    )


def test_chebx():
    torch.testing.assert_close(
        chebx,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )


def test_chebzero():
    torch.testing.assert_close(
        chebzero,
        torch.tensor([0.0]),
        check_dtype=False,
    )


def test_herm2poly():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 2]),
        torch.tensor([-2.0, 0, 4]),
        torch.tensor([0.0, -12, 0, 8]),
        torch.tensor([12.0, 0, -48, 0, 16]),
        torch.tensor([0.0, 120, 0, -160, 0, 32]),
        torch.tensor([-120.0, 0, 720, 0, -480, 0, 64]),
        torch.tensor([0.0, -1680, 0, 3360, 0, -1344, 0, 128]),
        torch.tensor([1680.0, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        torch.tensor([0.0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            herm2poly(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_hermadd():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            torch.testing.assert_close(
                hermtrim(
                    hermadd(
                        torch.tensor([0.0] * j + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
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
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        hermcompanion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = hermcompanion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = hermcompanion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.25


def test_hermder():
    with pytest.raises(TypeError):
        hermder(torch.tensor([0]), 0.5)

    with pytest.raises(ValueError):
        hermder(torch.tensor([0]), -1)

    for i in range(5):
        torch.testing.assert_close(
            hermtrim(
                hermder(
                    torch.tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            hermtrim(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                hermtrim(
                    hermder(
                        hermint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                hermtrim(
                    hermder(
                        hermint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        hermder(c2d, axis=0),
        torch.vstack([hermder(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        hermder(
            c2d,
            axis=1,
        ),
        torch.vstack([hermder(c) for c in c2d]),
    )


def test_hermdiv():
    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            quotient, remainder = hermdiv(
                hermadd(
                    input,
                    other,
                ),
                input,
            )

            torch.testing.assert_close(
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
    torch.testing.assert_close(
        hermdomain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )


def test_herme2poly():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 1]),
        torch.tensor([0.0, -3, 0, 1]),
        torch.tensor([3.0, 0, -6, 0, 1]),
        torch.tensor([0.0, 15, 0, -10, 0, 1]),
        torch.tensor([-15.0, 0, 45, 0, -15, 0, 1]),
        torch.tensor([0.0, -105, 0, 105, 0, -21, 0, 1]),
        torch.tensor([105.0, 0, -420, 0, 210, 0, -28, 0, 1]),
        torch.tensor([0.0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            herme2poly(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_hermeadd():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            torch.testing.assert_close(
                hermetrim(
                    hermeadd(
                        torch.tensor([0.0] * j + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
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
        hermecompanion(torch.tensor([]))

    with pytest.raises(ValueError):
        hermecompanion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = hermecompanion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = hermecompanion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.5


def test_hermeder():
    pytest.raises(TypeError, hermeder, torch.tensor([0]), 0.5)
    pytest.raises(ValueError, hermeder, torch.tensor([0]), -1)

    for i in range(5):
        torch.testing.assert_close(
            hermetrim(
                hermeder(torch.tensor([0.0] * i + [1.0]), order=0),
                tol=0.000001,
            ),
            hermetrim(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                hermetrim(
                    hermeder(
                        hermeint(torch.tensor([0.0] * i + [1.0]), order=j),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                hermetrim(
                    hermeder(
                        hermeint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        hermeder(c2d, axis=0),
        torch.vstack([hermeder(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        hermeder(
            c2d,
            axis=1,
        ),
        torch.vstack([hermeder(c) for c in c2d]),
    )


def test_hermediv():
    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            quotient, remainder = hermediv(
                hermeadd(
                    input,
                    other,
                ),
                input,
            )

            torch.testing.assert_close(
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
    torch.testing.assert_close(
        hermedomain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        hermefit(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        hermefit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    torch.testing.assert_close(
        hermefit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        hermefit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        hermefit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        hermefit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        hermefit(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        hermefit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermefit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    # assert_close(
    #     hermefit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([1]),
    #     ),
    #     tensor([0, 1]),
    # )

    # assert_close(
    #     hermefit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([0, 1]),
    #     ),
    #     tensor([0, 1]),
    # )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        hermeval(
            input,
            hermefit(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        hermefit(
            input,
            other,
            degree=4,
        ),
        hermefit(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )


def test_hermegauss():
    x, w = hermegauss(100)

    v = hermevander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, torch.eye(100))

    target = math.sqrt(2 * math.pi)
    torch.testing.assert_close(
        w.sum(),
        torch.tensor(target),
    )


def test_hermegrid2d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        hermegrid2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
            ),
        ),
        torch.einsum(
            "i,j->ij",
            x,
            y,
        ),
    )

    output = hermegrid2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([4.0, 2.0, 3.0]),
            torch.tensor([4.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 2


def test_hermegrid3d():
    c1d = torch.tensor([4.0, 2.0, 3.0])
    c3d = torch.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    torch.testing.assert_close(
        hermegrid3d(a, b, x3, c3d),
        torch.einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = torch.ones([2, 3])
    res = hermegrid3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3


def test_hermeint():
    pytest.raises(TypeError, hermeint, torch.tensor([0]), 0.5)
    pytest.raises(ValueError, hermeint, torch.tensor([0]), -1)
    pytest.raises(ValueError, hermeint, torch.tensor([0]), 1, [0, 0])
    pytest.raises(ValueError, hermeint, torch.tensor([0]), lower_bound=[0])
    pytest.raises(ValueError, hermeint, torch.tensor([0]), scale=[0])
    pytest.raises(TypeError, hermeint, torch.tensor([0]), axis=0.5)

    for i in range(2, 5):
        torch.testing.assert_close(
            hermetrim(
                hermeint(
                    torch.tensor([0.0]),
                    order=i,
                    k=([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            hermetrim(
                herme2poly(
                    hermeint(
                        poly2herme(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    ),
                ),
                tol=0.000001,
            ),
            hermetrim(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            hermeval(
                torch.tensor([-1]),
                hermeint(
                    poly2herme(
                        torch.tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            hermetrim(
                herme2poly(
                    hermeint(
                        poly2herme(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    ),
                ),
                tol=0.000001,
            ),
            hermetrim(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = hermeint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                hermetrim(
                    hermeint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
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
            target = torch.tensor([0.0] * i + [1.0])[:]
            for k in range(j):
                target = hermeint(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                hermetrim(
                    hermeint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = hermeint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                hermetrim(
                    hermeint(
                        torch.tensor([0.0] * i + [1.0]),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = hermeint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                hermetrim(
                    hermeint(
                        torch.tensor([0.0] * i + [1.0]),
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

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        hermeint(
            c2d,
            axis=0,
        ),
        torch.vstack([hermeint(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        hermeint(
            c2d,
            axis=1,
        ),
        torch.vstack([hermeint(c) for c in c2d]),
    )

    torch.testing.assert_close(
        hermeint(
            c2d,
            k=3,
            axis=1,
        ),
        torch.vstack([hermeint(c, k=3) for c in c2d]),
    )


def test_hermeline():
    torch.testing.assert_close(
        hermeline(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )


def test_hermemul():
    for index in range(5):
        input = torch.linspace(-3, 3, 100)

        val1 = hermeval(
            input,
            torch.tensor([0.0] * index + [1.0]),
        )

        for k in range(5):
            val2 = hermeval(
                input,
                torch.tensor([0.0] * k + [1.0]),
            )

            torch.testing.assert_close(
                hermeval(
                    input,
                    hermemul(
                        torch.tensor([0.0] * index + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
                    ),
                ),
                val1 * val2,
            )


def test_hermemulx():
    torch.testing.assert_close(
        hermetrim(
            hermemulx(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )
    torch.testing.assert_close(
        hermetrim(
            hermemulx(
                torch.tensor([1.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0, 1.0]),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            hermetrim(
                hermemulx(
                    torch.tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0] * (index - 1) + [index, 0.0, 1.0]),
        )


def test_hermeone():
    torch.testing.assert_close(
        hermeone,
        torch.tensor([1.0]),
        check_dtype=False,
    )


def test_hermepow():
    for j in range(5):
        for k in range(5):
            torch.testing.assert_close(
                hermetrim(
                    hermepow(
                        torch.arange(0.0, j + 1),
                        k,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    functools.reduce(
                        hermemul,
                        [torch.arange(0.0, j + 1)] * k,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermeroots():
    torch.testing.assert_close(
        hermeroots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        hermeroots(
            torch.tensor([1.0, 1.0]),
        ),
        torch.tensor([-1.0]),
    )

    for index in range(2, 5):
        input = torch.linspace(-1, 1, index)

        torch.testing.assert_close(
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
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] - 1

            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            torch.testing.assert_close(
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
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            -1,
        )

    torch.testing.assert_close(
        hermetrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        hermetrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        hermetrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            2,
        ),
        torch.tensor([0.0]),
    )


def test_hermeval():
    hermecoefficients = [
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([-1, 0, 1]),
        torch.tensor([0, -3, 0, 1]),
        torch.tensor([3, 0, -6, 0, 1]),
        torch.tensor([0, 15, 0, -10, 0, 1]),
        torch.tensor([-15, 0, 45, 0, -15, 0, 1]),
        torch.tensor([0, -105, 0, 105, 0, -21, 0, 1]),
        torch.tensor([105, 0, -420, 0, 210, 0, -28, 0, 1]),
        torch.tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    output = hermeval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in hermecoefficients:
        ys = [
            *ys,
            polyval(
                torch.linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for i in range(10):
        torch.testing.assert_close(
            hermeval(
                torch.linspace(-1, 1, 50),
                torch.tensor([0.0] * i + [1.0]),
            ),
            ys[i],
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = hermeval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = hermeval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = hermeval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_hermeval2d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError):
        hermeval2d(
            a,
            b[:2],
            torch.einsum(
                "i,j->ij",
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
            ),
        )

    torch.testing.assert_close(
        hermeval2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
            ),
        ),
        x * y,
    )

    output = hermeval2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([4.0, 2.0, 3.0]),
            torch.tensor([4.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_hermeval3d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError):
        hermeval3d(
            a,
            b,
            c[:2],
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
            ),
        )

    torch.testing.assert_close(
        hermeval3d(
            a,
            b,
            c,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
                torch.tensor([4.0, 2.0, 3.0]),
            ),
        ),
        x * y * z,
    )

    output = hermeval3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([4.0, 2.0, 3.0]),
            torch.tensor([4.0, 2.0, 3.0]),
            torch.tensor([4.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_hermevander():
    x = torch.arange(3)
    v = hermevander(
        x,
        3,
    )
    assert v.shape == (3, 4)
    for i in range(4):
        coefficients = torch.tensor([0.0] * i + [1.0])
        torch.testing.assert_close(
            v[..., i],
            hermeval(x, coefficients),
        )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    v = hermevander(
        x,
        3,
    )
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coefficients = torch.tensor([0.0] * i + [1.0])
        torch.testing.assert_close(
            v[..., i],
            hermeval(x, coefficients),
        )


def test_hermevander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = hermevander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        hermeval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = hermevander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)


def test_hermevander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = hermevander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        hermeval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = hermevander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)


def test_hermeweight():
    torch.testing.assert_close(
        hermeweight(
            torch.linspace(-5, 5, 11),
        ),
        torch.exp(-0.5 * torch.linspace(-5, 5, 11) ** 2),
    )


def test_hermex():
    torch.testing.assert_close(
        hermex,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )


def test_hermezero():
    torch.testing.assert_close(
        hermezero,
        torch.tensor([0.0]),
        check_dtype=False,
    )


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        hermfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ],
        ).T,
    )

    torch.testing.assert_close(
        hermfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    torch.testing.assert_close(
        hermfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        hermfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        hermfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        hermfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        hermfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        hermfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    # assert_close(
    #     hermfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=1,
    #     ),
    #     tensor([0.0j, 0.5j]),
    # )

    # assert_close(
    #     hermfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([0, 1]),
    #     ),
    #     tensor([0, 0.5]),
    # )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        hermval(
            input,
            hermfit(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        hermfit(
            input,
            other,
            degree=4,
        ),
        hermfit(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )


def test_hermgauss():
    x, w = hermgauss(100)

    v = hermvander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(
        vv,
        torch.eye(100),
    )

    torch.testing.assert_close(
        w.sum(),
        torch.tensor(math.sqrt(math.pi)),
    )


def test_hermgrid2d():
    c1d = torch.tensor([2.5, 1.0, 0.75])
    c2d = torch.einsum("i,j->ij", c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    target = torch.einsum("i,j->ij", y1, y2)
    torch.testing.assert_close(
        hermgrid2d(
            a,
            b,
            c2d,
        ),
        target,
    )

    z = torch.ones([2, 3])
    assert (
        hermgrid2d(
            z,
            z,
            c2d,
        ).shape
        == (2, 3) * 2
    )


def test_hermgrid3d():
    c1d = torch.tensor([2.5, 1.0, 0.75])
    c3d = torch.einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    torch.testing.assert_close(
        hermgrid3d(
            a,
            b,
            x3,
            c3d,
        ),
        torch.einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    z = torch.ones([2, 3])

    assert hermgrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_hermint():
    with pytest.raises(TypeError):
        hermint(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        hermint(
            torch.tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        hermint(
            torch.tensor([0]),
            order=1,
            k=torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        hermint(
            torch.tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        hermint(
            torch.tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        hermint(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            hermtrim(
                hermint(
                    torch.tensor([0.0]),
                    order=i,
                    k=([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0, 0.5]),
        )

    for i in range(5):
        torch.testing.assert_close(
            hermtrim(
                herm2poly(
                    hermint(
                        poly2herm(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            hermtrim(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            hermval(
                torch.tensor([-1.0]),
                hermint(
                    poly2herm(
                        torch.tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            hermtrim(
                herm2poly(
                    hermint(
                        poly2herm(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    ),
                ),
                tol=0.000001,
            ),
            hermtrim(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = hermint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                hermtrim(
                    hermint(
                        torch.tensor([0.0] * i + [1.0]),
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
            pol = torch.tensor([0.0] * i + [1.0])

            target = pol[:]

            for k in range(j):
                target = hermint(target, order=1, k=[k])

            torch.testing.assert_close(
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
            pol = torch.tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = hermint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
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

    c2d = torch.rand(3, 4)

    target = torch.vstack([hermint(c) for c in c2d.T]).T

    torch.testing.assert_close(
        hermint(
            c2d,
            axis=0,
        ),
        target,
    )

    target = torch.vstack([hermint(c) for c in c2d])

    torch.testing.assert_close(
        hermint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = torch.vstack([hermint(c, k=3) for c in c2d])

    torch.testing.assert_close(
        hermint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )


def test_hermline():
    torch.testing.assert_close(
        hermline(3, 4),
        torch.tensor([3.0, 2.0]),
    )


def test_hermmul():
    for i in range(5):
        input = torch.linspace(-3, 3, 100)

        val1 = hermval(
            input,
            torch.tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            val2 = hermval(
                input,
                torch.tensor([0.0] * j + [1.0]),
            )

            torch.testing.assert_close(
                hermval(
                    input,
                    hermmul(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                ),
                val1 * val2,
            )


def test_hermmulx():
    torch.testing.assert_close(
        hermtrim(
            hermmulx(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        hermmulx(
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0, 0.5]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            hermmulx(
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor([0.0] * (i - 1) + [i, 0.0, 0.5]),
        )


def test_hermone():
    torch.testing.assert_close(
        hermone,
        torch.tensor([1.0]),
        check_dtype=False,
    )


def test_hermpow():
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(
                hermtrim(
                    hermpow(
                        torch.arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                hermtrim(
                    functools.reduce(
                        hermmul,
                        [torch.arange(0.0, i + 1)] * j,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermroots():
    torch.testing.assert_close(
        hermroots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        hermroots(
            torch.tensor([1.0, 1.0]),
        ),
        torch.tensor([-0.5]),
    )

    for i in range(2, 5):
        input = torch.linspace(-1, 1, i)

        torch.testing.assert_close(
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
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            torch.testing.assert_close(
                hermtrim(
                    hermsub(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        hermtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        hermtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        hermtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )


def test_hermval():
    hermcoefficients = [
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([-2, 0, 4]),
        torch.tensor([0, -12, 0, 8]),
        torch.tensor([12, 0, -48, 0, 16]),
        torch.tensor([0, 120, 0, -160, 0, 32]),
        torch.tensor([-120, 0, 720, 0, -480, 0, 64]),
        torch.tensor([0, -1680, 0, 3360, 0, -1344, 0, 128]),
        torch.tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        torch.tensor([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    output = hermval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    input = torch.linspace(-1, 1, 50)

    for coefficient in hermcoefficients:
        ys = [
            *ys,
            polyval(
                input,
                coefficient,
            ),
        ]

    for index in range(10):
        torch.testing.assert_close(
            hermval(
                input,
                torch.tensor([0.0] * index + [1.0]),
            ),
            ys[index],
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = hermval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = hermval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = hermval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_hermval2d():
    c1d = torch.tensor([2.5, 1.0, 0.75])
    c2d = torch.einsum("i,j->ij", c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        hermval2d,
        a,
        b[:2],
        c2d,
    )

    torch.testing.assert_close(
        hermval2d(
            a,
            b,
            c2d,
        ),
        y1 * y2,
    )

    z = torch.ones([2, 3])
    res = hermval2d(
        z,
        z,
        c2d,
    )
    assert res.shape == (2, 3)


def test_hermval3d():
    c1d = torch.tensor([2.5, 1.0, 0.75])
    c3d = torch.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(ValueError, hermval3d, a, b, x3[:2], c3d)

    target = y1 * y2 * y3
    torch.testing.assert_close(
        hermval3d(a, b, x3, c3d),
        target,
    )

    z = torch.ones([2, 3])
    assert hermval3d(z, z, z, c3d).shape == (2, 3)


def test_hermvander():
    x = torch.arange(3)

    output = hermvander(
        x,
        degree=3,
    )

    assert output.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            hermval(
                x,
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

    output = hermvander(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=3,
    )

    assert output.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            hermval(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([0.0] * index + [1.0]),
            ),
        )


def test_hermvander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = hermvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        hermval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = hermvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)


def test_hermvander3d():
    a, b, x3 = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = hermvander3d(
        a,
        b,
        x3,
        degree=torch.tensor([1, 2, 3]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        hermval3d(a, b, x3, coefficients),
    )

    output = hermvander3d(
        a,
        b,
        x3,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)


def test_hermweight():
    torch.testing.assert_close(
        hermweight(torch.linspace(-5, 5, 11)),
        torch.exp(-(torch.linspace(-5, 5, 11) ** 2)),
    )


def test_hermx():
    torch.testing.assert_close(
        hermx,
        torch.tensor([0, 0.5]),
        check_dtype=False,
    )


def test_hermzero():
    torch.testing.assert_close(
        hermzero,
        torch.tensor([0.0]),
        check_dtype=False,
    )


def test_lag2poly():
    coefficients = [
        torch.tensor([1.0]) / 1,
        torch.tensor([1.0, -1]) / 1,
        torch.tensor([2.0, -4, 1]) / 2,
        torch.tensor([6.0, -18, 9, -1]) / 6,
        torch.tensor([24.0, -96, 72, -16, 1]) / 24,
        torch.tensor([120.0, -600, 600, -200, 25, -1]) / 120,
        torch.tensor([720.0, -4320, 5400, -2400, 450, -36, 1]) / 720,
    ]

    for index in range(7):
        torch.testing.assert_close(
            lag2poly(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_lagadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            torch.testing.assert_close(
                lagtrim(
                    lagadd(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        lagcompanion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = lagcompanion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = lagcompanion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == 1.5


def test_lagder():
    with pytest.raises(TypeError):
        lagder(
            torch.tensor([0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        lagder(
            torch.tensor([0]),
            order=-1,
        )

    for i in range(5):
        torch.testing.assert_close(
            lagtrim(
                lagder(
                    torch.tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            lagtrim(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                lagtrim(
                    lagder(
                        lagint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    torch.tensor([0.0] * i + [1.0]),
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
                    torch.tensor([0.0] * i + [1.0]),
                    torch.tensor([0.0] * j + [1.0]),
                ),
                torch.tensor([0.0] * i + [1.0]),
            )

            torch.testing.assert_close(
                lagtrim(
                    lagadd(
                        lagmul(
                            quotient,
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    lagadd(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagdomain():
    torch.testing.assert_close(
        lagdomain,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        lagval(
            input,
            lagfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        lagfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        lagfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    torch.testing.assert_close(
        lagfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        lagfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        lagfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        lagfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        lagfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ],
        ).T,
    )

    torch.testing.assert_close(
        lagfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                lagfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    # assert_close(
    #     lagfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([1]),
    #     ),
    #     tensor([1, -1]),
    # )

    # assert_close(
    #     lagfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([0, 1]),
    #     ),
    #     tensor([1, -1]),
    # )


def test_laggrid2d():
    c1d = torch.tensor([9.0, -14.0, 6.0])
    c2d = torch.einsum("i,j->ij", c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    torch.testing.assert_close(
        laggrid2d(
            a,
            b,
            c2d,
        ),
        torch.einsum("i,j->ij", y1, y2),
    )

    z = torch.ones([2, 3])
    assert (
        laggrid2d(
            z,
            z,
            c2d,
        ).shape
        == (2, 3) * 2
    )


def test_laggrid3d():
    c1d = torch.tensor([9.0, -14.0, 6.0])
    c3d = torch.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    target = torch.einsum("i,j,k->ijk", y1, y2, y3)
    torch.testing.assert_close(laggrid3d(a, b, x3, c3d), target)

    z = torch.ones([2, 3])
    assert laggrid3d(z, z, z, c3d).shape == (2, 3) * 3


def test_lagint():
    with pytest.raises(TypeError):
        lagint(
            torch.tensor([0]),
            0.5,
        )

    with pytest.raises(ValueError):
        lagint(
            torch.tensor([0]),
            -1,
        )

    with pytest.raises(ValueError):
        lagint(
            torch.tensor([0]),
            1,
            torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        lagint(
            torch.tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        lagint(
            torch.tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        lagint(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            lagtrim(
                lagint(
                    torch.tensor([0.0]),
                    order=i,
                    k=([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([1.0, -1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            lagtrim(
                lag2poly(
                    lagint(
                        poly2lag(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    ),
                ),
                tol=0.000001,
            ),
            lagtrim(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            lagval(
                torch.tensor([-1.0]),
                lagint(
                    poly2lag(
                        torch.tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            lagtrim(
                lag2poly(
                    lagint(
                        poly2lag(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    ),
                ),
                tol=0.000001,
            ),
            lagtrim(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = lagint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                lagtrim(
                    lagint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = lagint(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                lagtrim(
                    lagint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = lagint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                lagtrim(
                    lagint(
                        torch.tensor([0.0] * i + [1.0]),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = lagint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                lagtrim(
                    lagint(
                        torch.tensor([0.0] * i + [1.0]),
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

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        lagint(
            c2d,
            axis=0,
        ),
        torch.vstack([lagint(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        lagint(
            c2d,
            axis=1,
        ),
        torch.vstack([lagint(c) for c in c2d]),
    )

    torch.testing.assert_close(
        lagint(
            c2d,
            k=3,
            axis=1,
        ),
        torch.vstack([lagint(c, k=3) for c in c2d]),
    )


def test_lagline():
    torch.testing.assert_close(
        lagline(3.0, 4.0),
        torch.tensor([7.0, -4.0]),
    )


def test_lagmul():
    for i in range(5):
        input = torch.linspace(-3, 3, 100)

        a = lagval(
            input,
            torch.tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = lagval(
                input,
                torch.tensor([0.0] * j + [1.0]),
            )

            torch.testing.assert_close(
                lagval(
                    input,
                    lagtrim(
                        lagmul(
                            torch.tensor([0.0] * i + [1.0]),
                            torch.tensor([0.0] * j + [1.0]),
                        ),
                    ),
                ),
                a * b,
            )


def test_lagmulx():
    torch.testing.assert_close(
        lagtrim(
            lagmulx(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        lagtrim(
            lagmulx(
                torch.tensor([1.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0, -1.0]),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            lagtrim(
                lagmulx(
                    torch.tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            lagtrim(
                torch.tensor(
                    [0.0] * (index - 1) + [-index, 2.0 * index + 1.0, -(index + 1.0)]
                ),
                tol=0.000001,
            ),
        )


def test_lagone():
    torch.testing.assert_close(
        lagone,
        torch.tensor([1.0]),
        check_dtype=False,
    )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(
                lagtrim(
                    lagpow(
                        torch.arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    functools.reduce(
                        lagmul,
                        [torch.arange(0.0, i + 1)] * j,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagroots():
    torch.testing.assert_close(
        lagroots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        lagroots(
            torch.tensor([0.0, 1.0]),
        ),
        torch.tensor([1.0]),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            lagtrim(
                lagroots(
                    lagfromroots(
                        torch.linspace(0, 3, index),
                    ),
                ),
                tol=0.000001,
            ),
            lagtrim(
                torch.linspace(0, 3, index),
                tol=0.000001,
            ),
        )


def test_lagsub():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            torch.testing.assert_close(
                lagtrim(
                    lagsub(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        lagtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        lagtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        lagtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )


def test_lagval():
    lagcoefficients = [
        torch.tensor([1]) / 1,
        torch.tensor([1, -1]) / 1,
        torch.tensor([2, -4, 1]) / 2,
        torch.tensor([6, -18, 9, -1]) / 6,
        torch.tensor([24, -96, 72, -16, 1]) / 24,
        torch.tensor([120, -600, 600, -200, 25, -1]) / 120,
        torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720,
    ]

    output = lagval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )
    assert math.prod(output.shape) == 0

    ys = []

    input = torch.linspace(-1, 1, 50)

    for coefficient in lagcoefficients:
        ys = [
            *ys,
            polyval(
                input,
                coefficient,
            ),
        ]

    for i in range(7):
        torch.testing.assert_close(
            lagval(
                input,
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor(torch.tensor(ys[i])),
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = lagval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = lagval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = lagval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_lagval2d():
    c1d = torch.tensor([9.0, -14.0, 6.0])
    c2d = torch.einsum("i,j->ij", c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

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
    torch.testing.assert_close(
        lagval2d(
            a,
            b,
            c2d,
        ),
        target,
    )

    z = torch.ones([2, 3])
    assert lagval2d(
        z,
        z,
        c2d,
    ).shape == (2, 3)


def test_lagval3d():
    c1d = torch.tensor([9.0, -14.0, 6.0])
    c3d = torch.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, lagval3d, a, b, x3[:2], c3d)

    torch.testing.assert_close(
        lagval3d(
            a,
            b,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    assert lagval3d(
        torch.ones([2, 3]), torch.ones([2, 3]), torch.ones([2, 3]), c3d
    ).shape == (2, 3)


def test_lagvander():
    x = torch.arange(3)

    v = lagvander(x, 3)

    assert v.shape == (3, 4)

    for i in range(4):
        torch.testing.assert_close(
            v[..., i],
            lagval(
                x,
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    v = lagvander(x, 3)

    assert v.shape == (3, 2, 4)

    for i in range(4):
        torch.testing.assert_close(
            v[..., i],
            lagval(
                x,
                torch.tensor([0.0] * i + [1.0]),
            ),
        )


def test_lagvander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = lagvander2d(
        a,
        b,
        torch.tensor([1, 2]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        lagval2d(a, b, coefficients),
    )

    output = lagvander2d(
        a,
        b,
        torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)


def test_lagvander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = lagvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )
    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        lagval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = lagvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)


def test_lagweight():
    torch.testing.assert_close(
        lagweight(torch.linspace(0, 10, 11)),
        torch.exp(-torch.linspace(0, 10, 11)),
    )


def test_lagx():
    torch.testing.assert_close(
        lagx,
        torch.tensor([1.0, -1.0]),
        check_dtype=False,
    )


def test_lagzero():
    torch.testing.assert_close(
        lagzero,
        torch.tensor([0.0]),
        check_dtype=False,
    )


def test_leg2poly():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0, 3.0]) / 2.0,
        torch.tensor([0.0, -3.0, 0.0, 5.0]) / 2.0,
        torch.tensor([3.0, 0.0, -30, 0, 35]) / 8,
        torch.tensor([0.0, 15.0, 0, -70, 0, 63]) / 8,
        torch.tensor([-5.0, 0.0, 105, 0, -315, 0, 231]) / 16,
        torch.tensor([0.0, -35.0, 0, 315, 0, -693, 0, 429]) / 16,
        torch.tensor([35.0, 0.0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        torch.tensor([0.0, 315.0, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    for index in range(10):
        torch.testing.assert_close(
            leg2poly(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )


def test_legadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            torch.testing.assert_close(
                legtrim(
                    legadd(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
        legcompanion(torch.tensor([]))

    with pytest.raises(ValueError):
        legcompanion(torch.tensor([1]))

    for index in range(1, 5):
        output = legcompanion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    assert legcompanion(torch.tensor([1, 2]))[0, 0] == -0.5


def test_legdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = legdiv(
                legadd(
                    torch.tensor([0.0] * i + [1.0]),
                    torch.tensor([0.0] * j + [1.0]),
                ),
                torch.tensor([0.0] * i + [1.0]),
            )

            torch.testing.assert_close(
                legtrim(
                    legadd(
                        legmul(
                            quotient,
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    legadd(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legdomain():
    torch.testing.assert_close(
        legdomain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )


def test_legfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        legfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        legfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    torch.testing.assert_close(
        legfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        legfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        legfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        legfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        legfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        legfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    legfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    # assert_close(
    #     legfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([1]),
    #     ),
    #     tensor([0, 1]),
    # )

    # assert_close(
    #     legfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         degree=tensor([0, 1]),
    #     ),
    #     tensor([0, 1]),
    # )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        legval(
            input,
            legfit(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        legfit(
            input,
            other,
            degree=4,
        ),
        legfit(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )


def test_leggrid2d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        leggrid2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.0, 2.0, 2.0]),
                torch.tensor([2.0, 2.0, 2.0]),
            ),
        ),
        torch.einsum(
            "i,j->ij",
            x,
            y,
        ),
    )

    output = leggrid2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        ),
    )

    assert output.shape == (2, 3) * 2


def test_leggrid3d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        leggrid3d(
            a,
            b,
            c,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([2.0, 2.0, 2.0]),
                torch.tensor([2.0, 2.0, 2.0]),
                torch.tensor([2.0, 2.0, 2.0]),
            ),
        ),
        torch.einsum(
            "i,j,k->ijk",
            x,
            y,
            z,
        ),
    )

    output = leggrid3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_legint():
    with pytest.raises(TypeError):
        legint(
            torch.tensor([0]),
            0.5,
        )

    with pytest.raises(ValueError):
        legint(
            torch.tensor([0]),
            -1,
        )

    with pytest.raises(ValueError):
        legint(
            torch.tensor([0]),
            1,
            torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        legint(
            torch.tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        legint(
            torch.tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        legint(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        output = legint(
            torch.tensor([0.0]),
            order=i,
            k=[0.0] * (i - 2) + [1.0],
        )
        torch.testing.assert_close(
            legtrim(
                output,
                tol=0.000001,
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            legtrim(
                torch.tensor([i] + [0.0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            legval(
                torch.tensor([-1]),
                legint(
                    poly2leg(
                        torch.tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            legtrim(
                leg2poly(
                    legint(
                        poly2leg(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    )
                ),
                tol=0.000001,
            ),
            legtrim(
                torch.tensor([i] + [0.0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = legint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                legtrim(
                    legint(
                        torch.tensor([0.0] * i + [1.0]),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                legtrim(
                    legint(
                        torch.tensor([0.0] * i + [1.0]),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                legtrim(
                    legint(
                        torch.tensor([0.0] * i + [1.0]),
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
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = legint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                legtrim(
                    legint(
                        torch.tensor([0.0] * i + [1.0]),
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

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        legint(c2d, axis=0),
        torch.vstack([legint(c) for c in c2d.T]).T,
    )

    target = [legint(c) for c in c2d]

    target = torch.vstack(target)

    torch.testing.assert_close(
        legint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = [legint(c, k=3) for c in c2d]

    target = torch.vstack(target)

    torch.testing.assert_close(
        legint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )

    torch.testing.assert_close(
        legint(
            torch.tensor([1, 2, 3]),
            order=0,
        ),
        torch.tensor([1, 2, 3]),
    )


def test_legline():
    torch.testing.assert_close(
        legline(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )

    torch.testing.assert_close(
        legtrim(
            legline(3.0, 0.0),
            tol=0.000001,
        ),
        torch.tensor([3.0]),
    )


def test_legmul():
    for i in range(5):
        input = torch.linspace(-1, 1, 100)

        a = legval(
            input,
            torch.tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = legval(
                input,
                torch.tensor([0.0] * j + [1.0]),
            )

            torch.testing.assert_close(
                legval(
                    input,
                    legmul(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                ),
                a * b,
            )


def test_legmulx():
    torch.testing.assert_close(
        legtrim(
            legmulx(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        legtrim(
            legmulx(
                torch.tensor([1.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            legtrim(
                legmulx(
                    torch.tensor([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0] * (i - 1) + [i / (2 * i + 1), 0, (i + 1) / (2 * i + 1)]),
        )


def test_legone():
    torch.testing.assert_close(
        legone,
        torch.tensor([1.0]),
        check_dtype=False,
    )


def test_legpow():
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(
                legtrim(
                    legpow(
                        torch.arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    functools.reduce(
                        legmul,
                        [torch.arange(0.0, i + 1)] * j,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legroots():
    torch.testing.assert_close(
        legroots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        legroots(torch.tensor([1.0, 2.0])),
        torch.tensor([-0.5]),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            legtrim(
                legroots(
                    legfromroots(
                        torch.linspace(-1, 1, index),
                    ),
                ),
                tol=0.000001,
            ),
            legtrim(
                torch.linspace(-1, 1, index),
                tol=0.000001,
            ),
        )


def test_legsub():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            torch.testing.assert_close(
                legtrim(
                    legsub(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        legtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        legtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        legtrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )


def test_legval():
    legcoefficients = [
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([-1, 0, 3]) / 2,
        torch.tensor([0, -3, 0, 5]) / 2,
        torch.tensor([3, 0, -30, 0, 35]) / 8,
        torch.tensor([0, 15, 0, -70, 0, 63]) / 8,
        torch.tensor([-5, 0, 105, 0, -315, 0, 231]) / 16,
        torch.tensor([0, -35, 0, 315, 0, -693, 0, 429]) / 16,
        torch.tensor([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        torch.tensor([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    output = legval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in legcoefficients:
        ys = [
            *ys,
            polyval(
                torch.linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for i in range(10):
        torch.testing.assert_close(
            legval(
                torch.linspace(-1, 1, 50),
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor(ys[i]),
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = legval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = legval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = legval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_legval2d():
    c1d = torch.tensor([2.0, 2.0, 2.0])
    c2d = torch.einsum("i,j->ij", c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    pytest.raises(
        ValueError,
        legval2d,
        a,
        b[:2],
        c2d,
    )

    torch.testing.assert_close(
        legval2d(
            a,
            b,
            c2d,
        ),
        y1 * y2,
    )

    z = torch.ones([2, 3])
    assert legval2d(
        z,
        z,
        c2d,
    ).shape == (2, 3)


def test_legval3d():
    c1d = torch.tensor([2.0, 2.0, 2.0])

    c3d = torch.einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = torch.rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(x, torch.tensor([1.0, 2.0, 3.0]))

    pytest.raises(ValueError, legval3d, a, b, x3[:2], c3d)

    torch.testing.assert_close(
        legval3d(
            a,
            b,
            x3,
            c3d,
        ),
        y1 * y2 * y3,
    )

    z = torch.ones([2, 3])

    assert legval3d(z, z, z, c3d).shape == (2, 3)


def test_legvander():
    x = torch.arange(3)

    v = legvander(
        x,
        degree=3,
    )

    assert v.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            legval(
                x,
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    v = legvander(
        x,
        degree=3,
    )

    assert v.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            legval(
                x,
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

    with pytest.raises(ValueError):
        legvander(
            torch.tensor([1, 2, 3]),
            -1,
        )


def test_legvander2d():
    a, b, x3 = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = legvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )
    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        legval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = legvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)


def test_legvander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    target = legval3d(
        a,
        b,
        c,
        coefficients,
    )

    output = legvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )
    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        target,
    )

    output = legvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)


def test_legx():
    torch.testing.assert_close(
        legx,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )


def test_legzero():
    torch.testing.assert_close(
        legzero,
        torch.tensor([0.0]),
        check_dtype=False,
    )


def test_poly2cheb():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 2]),
        torch.tensor([0.0, -3, 0, 4]),
        torch.tensor([1.0, 0, -8, 0, 8]),
        torch.tensor([0.0, 5, 0, -20, 0, 16]),
        torch.tensor([-1.0, 0, 18, 0, -48, 0, 32]),
        torch.tensor([0.0, -7, 0, 56, 0, -112, 0, 64]),
        torch.tensor([1.0, 0, -32, 0, 160, 0, -256, 0, 128]),
        torch.tensor([0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            poly2cheb(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )


def test_poly2herm():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 2]),
        torch.tensor([-2.0, 0, 4]),
        torch.tensor([0.0, -12, 0, 8]),
        torch.tensor([12.0, 0, -48, 0, 16]),
        torch.tensor([0.0, 120, 0, -160, 0, 32]),
        torch.tensor([-120.0, 0, 720, 0, -480, 0, 64]),
        torch.tensor([0.0, -1680, 0, 3360, 0, -1344, 0, 128]),
        torch.tensor([1680.0, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        torch.tensor([0.0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            hermtrim(
                poly2herm(
                    coefficients[index],
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0] * index + [1.0]),
        )


def test_poly2herme():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 1]),
        torch.tensor([0.0, -3, 0, 1]),
        torch.tensor([3.0, 0, -6, 0, 1]),
        torch.tensor([0.0, 15, 0, -10, 0, 1]),
        torch.tensor([-15.0, 0, 45, 0, -15, 0, 1]),
        torch.tensor([0.0, -105, 0, 105, 0, -21, 0, 1]),
        torch.tensor([105.0, 0, -420, 0, 210, 0, -28, 0, 1]),
        torch.tensor([0.0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            poly2herme(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )


def test_poly2lag():
    coefficients = [
        torch.tensor([1.0]) / 1.0,
        torch.tensor([1.0, -1.0]) / 1.0,
        torch.tensor([2.0, -4.0, 1.0]) / 2.0,
        torch.tensor([6.0, -18.0, 9.0, -1.0]) / 6.0,
        torch.tensor([24.0, -96.0, 72.0, -16.0, 1.0]) / 24.0,
        torch.tensor([120.0, -600.0, 600.0, -200.0, 25.0, -1.0]) / 120.0,
        torch.tensor([720.0, -4320.0, 5400.0, -2400.0, 450.0, -36.0, 1.0]) / 720.0,
    ]

    for index in range(7):
        torch.testing.assert_close(
            poly2lag(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )


def test_poly2leg():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 3]) / 2,
        torch.tensor([0.0, -3, 0, 5]) / 2,
        torch.tensor([3.0, 0, -30, 0, 35]) / 8,
        torch.tensor([0.0, 15, 0, -70, 0, 63]) / 8,
        torch.tensor([-5.0, 0, 105, 0, -315, 0, 231]) / 16,
        torch.tensor([0.0, -35, 0, 315, 0, -693, 0, 429]) / 16,
        torch.tensor([35.0, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        torch.tensor([0.0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    for index in range(10):
        torch.testing.assert_close(
            poly2leg(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )


def test_polyadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            torch.testing.assert_close(
                polytrim(
                    polyadd(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
        polycompanion(torch.tensor([]))

    with pytest.raises(ValueError):
        polycompanion(torch.tensor([1]))

    for i in range(1, 5):
        output = polycompanion(
            torch.tensor([0.0] * i + [1.0]),
        )

        assert output.shape == (i, i)

    output = polycompanion(
        torch.tensor([1, 2]),
    )

    assert output[0, 0] == -0.5


def test_polydiv():
    quotient, remainder = polydiv(
        torch.tensor([2.0]),
        torch.tensor([2.0]),
    )

    torch.testing.assert_close(
        quotient,
        torch.tensor([1.0]),
    )

    torch.testing.assert_close(
        remainder,
        torch.tensor([0.0]),
    )

    quotient, remainder = polydiv(
        torch.tensor([2.0, 2.0]),
        torch.tensor([2.0]),
    )

    torch.testing.assert_close(
        quotient,
        torch.tensor([1.0, 1.0]),
    )

    torch.testing.assert_close(
        remainder,
        torch.tensor([0.0]),
    )

    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0, 2.0])
            other = torch.tensor([0.0] * k + [1.0, 2.0])

            quotient, remainder = polydiv(
                polyadd(
                    input,
                    other,
                ),
                input,
            )

            torch.testing.assert_close(
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
    torch.testing.assert_close(
        polydomain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )


def test_polyfit():
    def f(x: Tensor) -> Tensor:
        return x * (x - 1) * (x - 2)

    def g(x: Tensor) -> Tensor:
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
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

    torch.testing.assert_close(
        polyval(
            input,
            polyfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        polyfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                polyfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                polyfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        polyfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                polyfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                polyfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    # weight = zeros_like(input)
    #
    # weight[1::2] = 1.0
    #
    # assert_close(
    #     polyfit(
    #         input,
    #         other.at[0::2].set(0),
    #         degree=3,
    #         weight=weight,
    #     ),
    #     polyfit(
    #         input,
    #         other,
    #         degree=tensor([0, 1, 2, 3]),
    #     ),
    # )

    # assert_close(
    #     polyfit(
    #         input,
    #         other.at[0::2].set(0),
    #         degree=tensor([0, 1, 2, 3]),
    #         weight=weight,
    #     ),
    #     polyfit(
    #         input,
    #         other,
    #         degree=tensor([0, 1, 2, 3]),
    #     ),
    # )
    #
    # assert_close(
    #     polyfit(
    #         input,
    #         tensor([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
    #         degree=3,
    #         weight=weight,
    #     ),
    #     tensor(
    #         [
    #             polyfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #             polyfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )
    #
    # assert_close(
    #     polyfit(
    #         input,
    #         tensor([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
    #         degree=tensor([0, 1, 2, 3]),
    #         weight=weight,
    #     ),
    #     tensor(
    #         [
    #             polyfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #             polyfit(
    #                 input,
    #                 other,
    #                 degree=tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )
    #
    # assert_close(
    #     polyfit(
    #         tensor([1, 1j, -1, -1j]),
    #         tensor([1, 1j, -1, -1j]),
    #         1,
    #     ),
    #     tensor([0, 1]),
    # )
    #
    # assert_close(
    #     polyfit(
    #         tensor([1, 1j, -1, -0 - 1j]),
    #         tensor([1, 1j, -1, -0 - 1j]),
    #         (0, 1),
    #     ),
    #     tensor([0, 1]),
    # )
    #
    # input = linspace(-1, 1, 50)
    #
    # other = g(input)
    #
    # assert_close(
    #     polyval(
    #         input,
    #         polyfit(
    #             input,
    #             other,
    #             degree=tensor([4]),
    #         ),
    #     ),
    #     other,
    # )
    #
    # assert_close(
    #     polyval(
    #         input,
    #         polyfit(
    #             input,
    #             other,
    #             degree=tensor([0, 2, 4]),
    #         ),
    #     ),
    #     other,
    # )
    #
    # assert_close(
    #     polyfit(
    #         input,
    #         other,
    #         degree=tensor([4]),
    #     ),
    #     polyfit(
    #         input,
    #         other,
    #         degree=tensor([0, 2, 4]),
    #     ),
    # )


def test_polyfromroots():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 2]),
        torch.tensor([0.0, -3, 0, 4]),
        torch.tensor([1.0, 0, -8, 0, 8]),
        torch.tensor([0.0, 5, 0, -20, 0, 16]),
        torch.tensor([-1.0, 0, 18, 0, -48, 0, 32]),
        torch.tensor([0.0, -7, 0, 56, 0, -112, 0, 64]),
        torch.tensor([1.0, 0, -32, 0, 160, 0, -256, 0, 128]),
        torch.tensor([0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
    ]

    torch.testing.assert_close(
        polytrim(
            polyfromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for index in range(1, 5):
        input = torch.linspace(-math.pi, 0.0, 2 * index + 1)

        input = input[1::2]

        input = torch.cos(input)

        output = polyfromroots(input) * 2 ** (index - 1)

        torch.testing.assert_close(
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
    x = torch.rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        polygrid2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        torch.einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    output = polygrid2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 2


def test_polygrid3d():
    x = torch.rand(3, 5) * 2 - 1

    y = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    a, b, x3 = x
    y1, y2, y3 = y

    torch.testing.assert_close(
        polygrid3d(
            a,
            b,
            x3,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        torch.einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    output = polygrid3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 3


def test_polyint():
    with pytest.raises(TypeError):
        polyint(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        polyint(
            torch.tensor([0.0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        polyint(
            torch.tensor([0.0]),
            order=1,
            k=[0, 0],
        )

    with pytest.raises(ValueError):
        polyint(
            torch.tensor([0.0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        polyint(
            torch.tensor([0.0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        polyint(
            torch.tensor([0.0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            polytrim(
                polyint(
                    torch.tensor([0.0]),
                    order=i,
                    k=[0.0] * (i - 2) + [1.0],
                ),
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            polytrim(
                polyint(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                ),
            ),
            polytrim(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1.0)]),
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            polyval(
                torch.tensor([-1.0]),
                polyint(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            polytrim(
                polyint(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                    scale=2,
                ),
            ),
            polytrim(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = polyint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                polytrim(
                    polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                    ),
                ),
                polytrim(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                polytrim(
                    polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                    ),
                ),
                polytrim(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                polytrim(
                    polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                ),
                polytrim(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = polyint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                polytrim(
                    polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                ),
                polytrim(
                    target,
                ),
            )

    # c2d = np.random.random((3, 6))
    #
    # assert_close(
    #     polyint(
    #         c2d,
    #         axis=0,
    #     ),
    #     vstack([polyint(c) for c in c2d.T]).T,
    # )
    #
    # assert_close(
    #     polyint(
    #         c2d,
    #         axis=1,
    #     ),
    #     vstack([polyint(c) for c in c2d]),
    # )
    #
    # assert_close(
    #     polyint(
    #         c2d,
    #         k=3,
    #         axis=1,
    #     ),
    #     vstack([polyint(c, k=3) for c in c2d]),
    # )


def test_polyline():
    torch.testing.assert_close(
        polyline(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )

    torch.testing.assert_close(
        polyline(3.0, 0.0),
        torch.tensor([3.0, 0.0]),
    )


def test_polymul():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(j + k + 1)

            target[j + k] = target[j + k] + 1

            torch.testing.assert_close(
                polytrim(
                    polymul(
                        torch.tensor([0.0] * j + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_polymulx():
    torch.testing.assert_close(
        polymulx(
            torch.tensor([0.0]),
        ),
        torch.tensor([0.0, 0.0]),
    )

    torch.testing.assert_close(
        polymulx(
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            polymulx(
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor([0.0] * (i + 1) + [1.0]),
        )


def test_polyone():
    torch.testing.assert_close(
        polyone,
        torch.tensor([1.0]),
        check_dtype=False,
    )


def test_polypow():
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(
                polytrim(
                    polypow(
                        torch.arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    functools.reduce(
                        polymul,
                        [torch.arange(0.0, i + 1)] * j,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_polyroots():
    torch.testing.assert_close(
        polyroots(torch.tensor([1.0])),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        polyroots(torch.tensor([1.0, 2.0])),
        torch.tensor([-0.5]),
    )

    for index in range(2, 5):
        input = torch.linspace(-1, 1, index)

        torch.testing.assert_close(
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
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] - 1

            torch.testing.assert_close(
                polytrim(
                    polysub(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
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
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )


def test_polyval():
    output = polyval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    y = []

    input = torch.linspace(-1, 1, 50)

    for index in range(5):
        y = [
            *y,
            input**index,
        ]

    for index in range(5):
        torch.testing.assert_close(
            polyval(
                input,
                torch.tensor([0.0] * index + [1.0]),
            ),
            y[index],
        )

    torch.testing.assert_close(
        polyval(
            input,
            torch.tensor([0, -1, 0, 1]),
        ),
        input * (input**2 - 1),
    )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = polyval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_polyval2d():
    x = torch.rand(3, 5) * 2 - 1

    a, b, c = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        polyval2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        y1 * y2,
    )

    output = polyval2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_polyval3d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = polyval(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        polyval3d(
            a,
            b,
            c,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        x * y * z,
    )

    output = polyval3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)


def test_polyvalfromroots():
    with pytest.raises(ValueError):
        polyvalfromroots(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            tensor=False,
        )

    output = polyvalfromroots(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (0,)

    output = polyvalfromroots(
        torch.tensor([]),
        torch.tensor([[1.0] * 5]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (5, 0)

    torch.testing.assert_close(
        polyvalfromroots(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0]),
    )

    output = polyvalfromroots(
        torch.tensor([1.0]),
        torch.ones([3, 3]),
    )

    assert output.shape == (3, 1)

    input = torch.linspace(-1, 1, 50)

    evaluations = []

    for i in range(5):
        evaluations = [*evaluations, input**i]

    for i in range(1, 5):
        target = evaluations[i]

        torch.testing.assert_close(
            polyvalfromroots(
                input,
                torch.tensor([0.0] * i),
            ),
            target,
        )

    torch.testing.assert_close(
        polyvalfromroots(
            input,
            torch.tensor([-1.0, 0.0, 1.0]),
        ),
        input * (input - 1.0) * (input + 1.0),
    )

    for i in range(3):
        shape = (2,) * i

        input = torch.zeros(shape)

        output = polyvalfromroots(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = polyvalfromroots(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = polyvalfromroots(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

    ptest = torch.tensor([15.0, 2.0, -16.0, -2.0, 1.0])

    r = polyroots(ptest)

    torch.testing.assert_close(
        polyval(
            input,
            ptest,
        ),
        polyvalfromroots(
            input,
            r,
        ),
    )

    x = torch.arange(-3, 2)

    r = torch.randint(-5, 5, (3, 5)).to(torch.float64)

    target = torch.empty(r.shape[1:])

    for j in range(math.prod(target.shape)):
        target[j] = polyvalfromroots(
            x[j],
            r[:, j],
        )

    torch.testing.assert_close(
        polyvalfromroots(
            x,
            r,
            tensor=False,
        ),
        target,
    )

    x = torch.vstack([x, 2 * x])

    target = torch.empty(r.shape[1:] + x.shape)

    for j in range(r.shape[1]):
        for k in range(x.shape[0]):
            target[j, k, :] = polyvalfromroots(
                x[k],
                r[:, j],
            )

    torch.testing.assert_close(
        polyvalfromroots(
            x,
            r,
            tensor=True,
        ),
        target,
    )


def test_polyvander():
    output = polyvander(
        torch.arange(3.0),
        degree=torch.tensor([3]),
    )

    assert output.shape == (3, 4)

    for i in range(4):
        torch.testing.assert_close(
            output[..., i],
            polyval(
                torch.arange(3),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    output = polyvander(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=torch.tensor([3]),
    )

    assert output.shape == (3, 2, 4)

    for i in range(4):
        torch.testing.assert_close(
            output[..., i],
            polyval(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    with pytest.raises(ValueError):
        polyvander(
            torch.arange(3),
            degree=torch.tensor([-1]),
        )


def test_polyvander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = polyvander2d(a, b, degree=torch.tensor([1, 2]))

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        polyval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = polyvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)


def test_polyvander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = polyvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        polyval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = polyvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    assert output.shape == (5, 24)


def test_polyx():
    torch.testing.assert_close(
        polyx,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )


def test_polyzero():
    torch.testing.assert_close(
        polyzero,
        torch.tensor([0.0]),
        check_dtype=False,
    )
