import functools
import math

import jax
import numpy
import pytest
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
    polyx,
    polyzero,
)
from jax.numpy import (
    arange,
    array,
    linspace,
    zeros,
)
from numpy.testing import (
    assert_array_almost_equal,
)
from torch import tensor

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
                array([2.0] + [1.0] * i, dtype=numpy.float64),
            ),
            array([0.5] * i + [2.0] + [0.5] * i, dtype=numpy.float64),
        )


def test__fit():
    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0]),
            degree=array([-1.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([[1.0]]),
            array([1.0]),
            degree=array([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([]),
            array([1.0]),
            degree=array([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([[[1.0]]]),
            degree=array([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1.0, 2.0]),
            array([1.0]),
            degree=array([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0, 2.0]),
            degree=array([0.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0]),
            degree=array([0.0]),
            weight=[[1.0]],
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0]),
            degree=array([0.0]),
            weight=array([1.0, 1.0]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0]),
            degree=array([-1.0]),
        )

    with pytest.raises(ValueError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0]),
            degree=array([2.0, -1.0, 6.0]),
        )

    with pytest.raises(TypeError):
        _fit(
            _vandermonde,
            array([1.0]),
            array([1.0]),
            degree=array([]),
        )


def test__get_domain():
    assert_array_almost_equal(
        _get_domain(
            array([1.0, 10.0, 3.0, -1.0]),
        ),
        array([-1.0, 10.0]),
    )

    assert_array_almost_equal(
        _get_domain(
            array([1.0 + 1j, 1.0 - 1j, 0.0, 2]),
        ),
        array([-1.0j, 2.0 + 1j]),
    )


def test__map_domain():
    assert_array_almost_equal(
        _map_domain(
            array([0.0, 4.0]),
            array([0.0, 4.0]),
            array([1.0, 3.0]),
        ),
        array([1.0, 3.0]),
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
            array([0.0, 4.0]),
            array([1.0, 3.0]),
        ),
        array([[1.0, 3.0], [1.0, 3.0]]),
    )


def test__map_parameters():
    assert_array_almost_equal(
        _map_parameters(
            array([0.0, 4.0]),
            array([1.0, 3.0]),
        ),
        array([1.0, 0.5]),
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
            array([1.0, 2.0, 3.0]),
            exponent=5,
            maximum_exponent=4,
        )


def test__trim_coefficients():
    with pytest.raises(ValueError):
        _trim_coefficients(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_array_almost_equal(
        _trim_coefficients(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        _trim_coefficients(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        _trim_coefficients(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        array([0.0]),
    )


def test__trim_sequence():
    for _ in range(5):
        assert_array_almost_equal(
            _trim_sequence(
                array([1.0] + [0.0] * 5),
            ),
            array([1.0]),
        )


def test__vandermonde():
    with pytest.raises(ValueError):
        _vandermonde(
            (),
            array([1.0, 2.0, 3.0]),
            array([90.0]),
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
                array([0.5] * i + [2.0] + [0.5] * i, dtype=numpy.float64),
            ),
            array([2.0] + [1.0] * i, dtype=numpy.float64),
        )


def test_chebadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_array_almost_equal(
                chebtrim(
                    chebadd(
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = chebdiv(
                chebadd(
                    array([0.0] * i + [1.0]),
                    array([0.0] * j + [1.0]),
                ),
                array([0.0] * i + [1.0]),
            )

            assert_array_almost_equal(
                chebtrim(
                    chebadd(
                        chebmul(
                            quotient,
                            array([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    chebadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_chebdomain():
    assert_array_almost_equal(
        chebdomain,
        tensor([-1.0, 1.0]),
    )


def test_chebline():
    torch.testing.assert_close(
        chebline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )


def test_chebmul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[abs(i + j)].set(target[abs(i + j)] + 0.5)
            target = target.at[abs(i - j)].set(target[abs(i - j)] + 0.5)

            assert_array_almost_equal(
                chebtrim(
                    chebmul(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            chebmulx(
                array([0.0]),
            ),
            tol=0.000001,
        ),
        array([0.0]),
    )

    assert_array_almost_equal(
        chebtrim(
            chebmulx(
                array([1.0]),
            ),
            tol=0.000001,
        ),
        array([0.0, 1.0]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            chebtrim(
                chebmulx(
                    array([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            array([0.0] * (i - 1) + [0.5, 0, 0.5]),
        )


def test_chebone():
    torch.testing.assert_close(
        chebone,
        tensor([1.0]),
    )


def test_chebpow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                chebtrim(
                    chebpow(
                        arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                chebtrim(
                    functools.reduce(
                        chebmul,
                        [arange(0.0, i + 1)] * j,
                        array([1.0]),
                    ),
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
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            array([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_array_almost_equal(
        chebtrim(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        chebtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        chebtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        array([0.0]),
    )


def test_chebval():
    output = chebval(
        array([]),
        array([1.0]),
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
        assert_array_almost_equal(
            chebval(
                linspace(-1, 1, 50),
                array([0.0] * index + [1.0]),
            ),
            array(ys[index]),
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = chebval(
            input,
            array([1.0]),
        )

        assert output.shape == shape

        output = chebval(
            input,
            array([1.0, 0.0]),
        )

        assert output.shape == shape

        output = chebval(
            input,
            array([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_chebx():
    torch.testing.assert_close(
        chebx,
        tensor([0.0, 1.0]),
    )


def test_chebzero():
    torch.testing.assert_close(
        chebzero,
        tensor([0.0]),
    )


def test_hermadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_array_almost_equal(
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
    torch.testing.assert_close(
        hermdomain,
        tensor([-1.0, 1.0]),
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
                        tensor([0.0] * i + [1.0]),
                        tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    target,
                    tol=0.000001,
                ),
            )


def test_hermediv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = hermediv(
                hermeadd(
                    array([0.0] * i + [1.0]),
                    array([0.0] * j + [1.0]),
                ),
                array([0.0] * i + [1.0]),
            )

            assert_array_almost_equal(
                hermetrim(
                    hermeadd(
                        hermemul(
                            quotient,
                            array([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    hermeadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_hermedomain():
    torch.testing.assert_close(
        hermedomain,
        tensor([-1.0, 1.0]),
    )


def test_hermeline():
    torch.testing.assert_close(
        hermeline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )


def test_hermemul():
    for i in range(5):
        val1 = hermeval(
            linspace(-3, 3, 100),
            array([0.0] * i + [1.0]),
        )

        for j in range(5):
            val2 = hermeval(
                linspace(-3, 3, 100),
                array([0.0] * j + [1.0]),
            )

            assert_array_almost_equal(
                hermeval(
                    linspace(-3, 3, 100),
                    hermemul(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                ),
                val1 * val2,
            )


def test_hermemulx():
    assert_array_almost_equal(
        hermetrim(
            hermemulx(
                array([0.0]),
            ),
            tol=0.000001,
        ),
        array([0.0]),
    )
    assert_array_almost_equal(
        hermetrim(
            hermemulx(
                array([1.0]),
            ),
            tol=0.000001,
        ),
        array([0.0, 1.0]),
    )
    for i in range(1, 5):
        assert_array_almost_equal(
            hermetrim(
                hermemulx(
                    array([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            array([0] * (i - 1) + [i, 0, 1]),
        )


def test_hermeone():
    torch.testing.assert_close(
        hermeone,
        tensor([1.0]),
    )


def test_hermepow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
                hermetrim(
                    hermepow(
                        arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                hermetrim(
                    functools.reduce(
                        hermemul,
                        array([arange(0.0, i + 1)] * j),
                        array([1.0]),
                    ),
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
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            array([2.0, -1.0, 1.0, 0.0]),
            -1,
        )

    assert_array_almost_equal(
        hermetrim(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        hermetrim(
            array([2.0, -1.0, 1.0, 0.0]),
            1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        hermetrim(
            array([2.0, -1.0, 1.0, 0.0]),
            2,
        ),
        array([0.0]),
    )


def test_hermeval():
    output = hermeval(
        array([]),
        array([1.0]),
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
        assert_array_almost_equal(
            hermeval(
                linspace(-1, 1, 50),
                array([0.0] * i + [1.0]),
            ),
            ys[i],
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = hermeval(
            input,
            array([1.0]),
        )

        assert output.shape == shape

        output = hermeval(
            input,
            array([1.0, 0.0]),
        )

        assert output.shape == shape

        output = hermeval(
            input,
            array([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_hermex():
    torch.testing.assert_close(
        hermex,
        tensor([0.0, 1.0]),
    )


def test_hermezero():
    torch.testing.assert_close(
        hermezero,
        tensor([0.0]),
    )


def test_hermline():
    torch.testing.assert_close(
        hermline(3, 4),
        tensor([3.0, 2.0]),
    )


def test_hermmul():
    for i in range(5):
        val1 = hermval(
            linspace(-3, 3, 100),
            array([0.0] * i + [1.0]),
        )

        for j in range(5):
            val2 = hermval(
                linspace(-3, 3, 100),
                array([0.0] * j + [1.0]),
            )

            assert_array_almost_equal(
                hermval(
                    linspace(-3, 3, 100),
                    hermmul(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                ),
                val1 * val2,
            )


def test_hermmulx():
    assert_array_almost_equal(
        hermtrim(
            hermmulx(
                array([0.0]),
            ),
            tol=0.000001,
        ),
        array([0.0]),
    )

    assert_array_almost_equal(
        hermmulx(
            array([1.0]),
        ),
        array([0.0, 0.5]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            hermmulx(
                array([0.0] * i + [1.0]),
            ),
            array([0.0] * (i - 1) + [i, 0.0, 0.5]),
        )


def test_hermone():
    torch.testing.assert_close(
        hermone,
        tensor([1.0]),
    )


def test_hermpow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
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
                        array([arange(0.0, i + 1)] * j),
                        array([1.0]),
                    ),
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
    with pytest.raises(ValueError):
        hermtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_array_almost_equal(
        hermtrim(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        hermtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        hermtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        array([0.0]),
    )


def test_hermval():
    output = hermval(
        array([]),
        array([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in hermcoefficients:
        ys = [
            *ys,
            polyval(
                linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for index in range(10):
        assert_array_almost_equal(
            hermval(
                linspace(-1, 1, 50),
                array([0] * index + [1]),
            ),
            ys[index],
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = hermval(
            input,
            array([1.0]),
        )

        assert output.shape == shape

        output = hermval(
            input,
            array([1.0, 0.0]),
        )

        assert output.shape == shape

        output = hermval(
            input,
            array([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_hermx():
    torch.testing.assert_close(
        hermx,
        tensor([0, 0.5]),
    )


def test_hermzero():
    torch.testing.assert_close(
        hermzero,
        tensor([0.0]),
    )


def test_lagadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_array_almost_equal(
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


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = lagdiv(
                lagadd(
                    array([0.0] * i + [1.0]),
                    array([0.0] * j + [1.0]),
                ),
                array([0.0] * i + [1.0]),
            )

            assert_array_almost_equal(
                lagtrim(
                    lagadd(
                        lagmul(
                            quotient,
                            array([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                lagtrim(
                    lagadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_lagdomain():
    torch.testing.assert_close(
        lagdomain,
        tensor([0.0, 1.0]),
    )


def test_lagline():
    torch.testing.assert_close(
        lagline(3.0, 4.0),
        tensor([7.0, -4.0]),
    )


def test_lagmul():
    for i in range(5):
        a = lagval(
            linspace(-3, 3, 100),
            array([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = lagval(
                linspace(-3, 3, 100),
                array([0.0] * j + [1.0]),
            )

            assert_array_almost_equal(
                lagval(
                    linspace(-3, 3, 100),
                    lagtrim(
                        lagmul(
                            array([0.0] * i + [1.0]),
                            array([0.0] * j + [1.0]),
                        ),
                    ),
                ),
                a * b,
            )


def test_lagmulx():
    assert_array_almost_equal(
        lagtrim(
            lagmulx(
                array([0.0]),
            ),
            tol=0.000001,
        ),
        array([0.0]),
    )

    assert_array_almost_equal(
        lagtrim(
            lagmulx(
                array([1.0]),
            ),
            tol=0.000001,
        ),
        array([1, -1]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            lagtrim(
                lagmulx(
                    array([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            lagtrim(
                array([0.0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]),
                tol=0.000001,
            ),
        )


def test_lagone():
    torch.testing.assert_close(
        lagone,
        tensor([1.0]),
    )


def test_lagpow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
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
                        array([1.0]),
                    ),
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
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            array([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_array_almost_equal(
        lagtrim(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        lagtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        lagtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        array([0.0]),
    )


def test_lagval():
    output = lagval(
        array([]),
        array([1.0]),
    )
    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in lagcoefficients:
        ys = [
            *ys,
            polyval(
                linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for i in range(7):
        assert_array_almost_equal(
            lagval(
                linspace(-1, 1, 50),
                array([0.0] * i + [1.0]),
            ),
            array(array(ys[i])),
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = lagval(
            input,
            array([1.0]),
        )

        assert output.shape == shape

        output = lagval(
            input,
            array([1.0, 0.0]),
        )

        assert output.shape == shape

        output = lagval(
            input,
            array([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_lagx():
    torch.testing.assert_close(
        lagx,
        tensor([1.0, -1.0]),
    )


def test_lagzero():
    torch.testing.assert_close(
        lagzero,
        tensor([0.0]),
    )


def test_legadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_array_almost_equal(
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


def test_legdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = legdiv(
                legadd(
                    array([0.0] * i + [1.0]),
                    array([0.0] * j + [1.0]),
                ),
                array([0.0] * i + [1.0]),
            )

            assert_array_almost_equal(
                legtrim(
                    legadd(
                        legmul(
                            quotient,
                            array([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                legtrim(
                    legadd(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )


def test_legdomain():
    torch.testing.assert_close(
        legdomain,
        tensor([-1.0, 1.0]),
    )


def test_legline():
    torch.testing.assert_close(
        legline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )

    assert_array_almost_equal(
        legtrim(
            legline(3.0, 0.0),
            tol=0.000001,
        ),
        tensor([3.0]),
    )


def test_legmul():
    for i in range(5):
        a = legval(
            linspace(-1, 1, 100),
            array([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = legval(
                linspace(-1, 1, 100),
                array([0.0] * j + [1.0]),
            )

            assert_array_almost_equal(
                legval(
                    linspace(-1, 1, 100),
                    legmul(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
                    ),
                ),
                a * b,
            )


def test_legmulx():
    assert_array_almost_equal(
        legtrim(
            legmulx(
                array([0.0]),
            ),
            tol=0.000001,
        ),
        array([0.0]),
    )

    assert_array_almost_equal(
        legtrim(
            legmulx(
                array([1.0]),
            ),
            tol=0.000001,
        ),
        array([0.0, 1.0]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            legtrim(
                legmulx(
                    array([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            array([0] * (i - 1) + [i / (2 * i + 1), 0, (i + 1) / (2 * i + 1)]),
        )


def test_legone():
    torch.testing.assert_close(
        legone,
        tensor([1.0]),
    )


def test_legpow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
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
                        array([arange(0.0, i + 1)] * j),
                        array([1.0]),
                    ),
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
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            array([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_array_almost_equal(
        legtrim(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        legtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        legtrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        array([0.0]),
    )


def test_legval():
    output = legval(
        array([]),
        array([1.0]),
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
        assert_array_almost_equal(
            legval(
                linspace(-1, 1, 50),
                array([0.0] * i + [1.0]),
            ),
            array(ys[i]),
        )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = legval(
            input,
            array([1.0]),
        )

        assert output.shape == shape

        output = legval(
            input,
            array([1.0, 0.0]),
        )

        assert output.shape == shape

        output = legval(
            input,
            array([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_legx():
    torch.testing.assert_close(
        legx,
        tensor([0.0, 1.0]),
    )


def test_legzero():
    torch.testing.assert_close(
        legzero,
        tensor([0.0]),
    )


def test_polyadd():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            assert_array_almost_equal(
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


def test_polydiv():
    quotient, remainder = polydiv(
        array([2]),
        array([2]),
    )

    assert_array_almost_equal(
        quotient,
        array([1.0]),
    )

    assert_array_almost_equal(
        remainder,
        array([0.0]),
    )

    quotient, remainder = polydiv(
        array([2, 2]),
        array([2]),
    )

    assert_array_almost_equal(
        quotient,
        array([1.0, 1.0]),
    )

    assert_array_almost_equal(
        remainder,
        array([0.0]),
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
    torch.testing.assert_close(
        polydomain,
        tensor([-1.0, 1.0]),
    )


def test_polyline():
    torch.testing.assert_close(
        polyline(3.0, 4.0),
        tensor([3.0, 4.0]),
    )

    torch.testing.assert_close(
        polyline(3.0, 0.0),
        tensor([3.0, 0.0]),
    )


def test_polymul():
    for i in range(5):
        for j in range(5):
            target = zeros(i + j + 1)

            target = target.at[i + j].set(target[i + j] + 1)

            assert_array_almost_equal(
                polytrim(
                    polymul(
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            array([0.0]),
        ),
        array([0, 0]),
    )

    assert_array_almost_equal(
        polymulx(
            array([1.0]),
        ),
        array([0.0, 1.0]),
    )

    for i in range(1, 5):
        assert_array_almost_equal(
            polymulx(
                array([0.0] * i + [1.0]),
            ),
            array([0.0] * (i + 1) + [1.0]),
        )


def test_polyone():
    torch.testing.assert_close(
        polyone,
        tensor([1.0]),
    )


def test_polypow():
    for i in range(5):
        for j in range(5):
            assert_array_almost_equal(
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
                        array([1.0]),
                    ),
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
                        array([0.0] * i + [1.0]),
                        array([0.0] * j + [1.0]),
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
            array([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    assert_array_almost_equal(
        polytrim(
            array([2.0, -1.0, 1.0, 0.0]),
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    assert_array_almost_equal(
        polytrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        array([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    assert_array_almost_equal(
        polytrim(
            array([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        array([0.0]),
    )


def test_polyval():
    output = polyval(
        array([]),
        array([1.0]),
    )

    assert math.prod(output.shape) == 0

    y = []

    for index in range(5):
        y = [
            *y,
            linspace(-1, 1, 50) ** index,
        ]

    for index in range(5):
        assert_array_almost_equal(
            polyval(
                linspace(-1, 1, 50),
                array([0.0] * index + [1.0]),
            ),
            y[index],
        )

    assert_array_almost_equal(
        polyval(
            linspace(-1, 1, 50),
            array([0, -1, 0, 1]),
        ),
        linspace(-1, 1, 50) * (linspace(-1, 1, 50) ** 2 - 1),
    )

    for index in range(3):
        shape = (2,) * index

        input = zeros(shape)

        output = polyval(
            input,
            array([1.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            array([1.0, 0.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            array([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape


def test_polyx():
    torch.testing.assert_close(
        polyx,
        tensor([0.0, 1.0]),
    )


def test_polyzero():
    torch.testing.assert_close(
        polyzero,
        tensor([0.0]),
    )
