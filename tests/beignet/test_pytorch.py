import functools
import math

import pytest
import torch
import torch.testing
from beignet.pytorch import (
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
    chebdiv,
    chebdomain,
    chebline,
    chebmul,
    chebmulx,
    chebone,
    chebpow,
    chebsub,
    chebtrim,
    chebval,
    chebx,
    chebzero,
    hermadd,
    hermdiv,
    hermdomain,
    hermeadd,
    hermediv,
    hermedomain,
    hermeline,
    hermemul,
    hermemulx,
    hermeone,
    hermepow,
    hermesub,
    hermetrim,
    hermeval,
    hermex,
    hermezero,
    hermline,
    hermmul,
    hermmulx,
    hermone,
    hermpow,
    hermsub,
    hermtrim,
    hermval,
    hermx,
    hermzero,
    lagadd,
    lagdiv,
    lagdomain,
    lagline,
    lagmul,
    lagmulx,
    lagone,
    lagpow,
    lagsub,
    lagtrim,
    lagval,
    lagx,
    lagzero,
    legadd,
    legdiv,
    legdomain,
    legline,
    legmul,
    legmulx,
    legone,
    legpow,
    legsub,
    legtrim,
    legval,
    legx,
    legzero,
    polyadd,
    polydiv,
    polydomain,
    polyline,
    polymul,
    polymulx,
    polyone,
    polypow,
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

torch.set_default_dtype(torch.float64)

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

lagcoefficients = [
    torch.tensor([1]) / 1,
    torch.tensor([1, -1]) / 1,
    torch.tensor([2, -4, 1]) / 2,
    torch.tensor([6, -18, 9, -1]) / 6,
    torch.tensor([24, -96, 72, -16, 1]) / 24,
    torch.tensor([120, -600, 600, -200, 25, -1]) / 120,
    torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720,
]

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

polycoefficients = [
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

    # torch.testing.assert_close(
    #     _map_domain(
    #         torch.tensor([-0 - 1j, 2 + 1j]),
    #         torch.tensor([-0 - 1j, 2 + 1j]),
    #         torch.tensor([-2.0, 2.0]),
    #     ),
    #     torch.tensor([-2.0, 2.0]),
    # )

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


def test_chebx():
    torch.testing.assert_close(
        chebx,
        torch.tensor([0.0, 1.0]),
    )


def test_chebzero():
    torch.testing.assert_close(
        chebzero,
        torch.tensor([0.0]),
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


def test_hermex():
    torch.testing.assert_close(
        hermex,
        torch.tensor([0.0, 1.0]),
    )


def test_hermezero():
    torch.testing.assert_close(
        hermezero,
        torch.tensor([0.0]),
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
                torch.tensor([0] * index + [1]),
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


def test_hermx():
    torch.testing.assert_close(
        hermx,
        torch.tensor([0, 0.5]),
    )


def test_hermzero():
    torch.testing.assert_close(
        hermzero,
        torch.tensor([0.0]),
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


def test_lagx():
    torch.testing.assert_close(
        lagx,
        torch.tensor([1.0, -1.0]),
    )


def test_lagzero():
    torch.testing.assert_close(
        lagzero,
        torch.tensor([0.0]),
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


def test_legx():
    torch.testing.assert_close(
        legx,
        torch.tensor([0.0, 1.0]),
    )


def test_legzero():
    torch.testing.assert_close(
        legzero,
        torch.tensor([0.0]),
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
    )


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

    x1, x2, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        polyval2d(
            x1,
            x2,
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
    x = torch.rand(3, 5) * 2 - 1

    x1, x2, x3 = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        polyval3d(
            x1,
            x2,
            x3,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        y1 * y2 * y3,
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
    pytest.raises(
        ValueError,
        polyvalfromroots,
        torch.tensor([1]),
        torch.tensor([1]),
        tensor=False,
    )

    output = polyvalfromroots(
        torch.tensor([]),
        torch.tensor([1]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (0,)

    output = polyvalfromroots(
        torch.tensor([]),
        torch.tensor([[1] * 5]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (5, 0)

    # assert_array_almost_equal(
    #     polyvalfromroots(
    #         array([1]),
    #         array([1]),
    #     ),
    #     array([0]),
    # )
    #
    # assert polyvalfromroots(array([1]), ones((3, 3))).shape == (3, 1)
    #
    # input = linspace(-1, 1, 50)
    #
    # y = [input**i for i in range(5)]
    #
    # for j in range(1, 5):
    #     target = y[j]
    #
    #     assert_array_almost_equal(
    #         polyvalfromroots(
    #             input,
    #             array([0] * j),
    #         ),
    #         target,
    #     )
    #
    # assert_array_almost_equal(
    #     polyvalfromroots(
    #         input,
    #         array([-1, 0, 1]),
    #     ),
    #     input * (input - 1) * (input + 1),
    # )
    #
    # for j in range(3):
    #     dims = (2,) * j
    #     x = zeros(dims)
    #     assert polyvalfromroots(x, array([1])).shape == dims
    #     assert polyvalfromroots(x, array([1, 0])).shape == dims
    #     assert polyvalfromroots(x, array([1, 0, 0])).shape == dims

    # ptest = array([15, 2, -16, -2, 1])
    #
    # r = polyroots(ptest)
    #
    # torch.testing.assert_close(
    #     polyval(input, ptest),
    #     polyvalfromroots(input, r),
    # )
    #
    # x = torch.arange(-3, 2)
    #
    # r = jax.random.randint(key, [3, 5], -5, 5)
    #
    # target = torch.empty(r.shape[1:])
    #
    # for j in range(target.size):
    #     target = target.at[j].set(polyvalfromroots(x[j], r[:, j]))
    #
    # torch.testing.assert_close(
    #     polyvalfromroots(x, r, tensor=False),
    #     target,
    # )
    #
    # x = torch.vstack([x, 2 * x])
    #
    # target = torch.empty(r.shape[1:] + x.shape)
    #
    # for j in range(r.shape[1]):
    #     for k in range(x.shape[0]):
    #         target[j, k, :] = polyvalfromroots(x[k], r[:, j])
    #
    # torch.testing.assert_close(
    #     polyvalfromroots(x, r, tensor=True),
    #     target,
    # )


def test_polyvander():
    output = polyvander(
        torch.arange(3),
        degree=torch.tensor(3),
    )

    assert output.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            polyval(
                torch.arange(3),
                torch.tensor([0] * index + [1]),
            ),
        )

    output = polyvander(
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        degree=torch.tensor(3),
    )

    assert output.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            polyval(
                torch.tensor([[1, 2], [3, 4], [5, 6]]),
                torch.tensor([0] * index + [1]),
            ),
        )

    with pytest.raises(ValueError):
        polyvander(
            torch.arange(3),
            torch.tensor([-1]),
        )


def test_polyvander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    torch.testing.assert_close(
        torch.dot(
            polyvander2d(
                a,
                b,
                degree=torch.tensor([1, 2]),
            ),
            torch.ravel(coefficients),
        ),
        polyval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = polyvander2d(
        torch.tensor([a]),
        torch.tensor([b]),
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (1, 5, 6)


def test_polyvander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    torch.testing.assert_close(
        torch.dot(
            polyvander3d(
                a,
                b,
                c,
                degree=torch.tensor([1.0, 2.0, 3.0]),
            ),
            torch.ravel(coefficients),
        ),
        polyval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = polyvander3d(
        torch.tensor([a]),
        torch.tensor([b]),
        torch.tensor([c]),
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    assert output.shape == (1, 5, 24)


def test_polyx():
    torch.testing.assert_close(
        polyx,
        torch.tensor([0.0, 1.0]),
    )


def test_polyzero():
    torch.testing.assert_close(
        polyzero,
        torch.tensor([0.0]),
    )
