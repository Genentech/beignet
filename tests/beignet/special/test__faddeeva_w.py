import math

import beignet.special
import hypothesis
import hypothesis.strategies
import scipy
import torch


@hypothesis.strategies.composite
def _strategy(function):
    dtype = function(hypothesis.strategies.sampled_from([torch.float64, torch.float32]))

    # avoid overflow of exp(y^2)
    limit = math.sqrt(math.log(torch.finfo(dtype).max)) - 0.2

    x = function(
        hypothesis.strategies.floats(
            min_value=-limit, max_value=limit, allow_nan=False, allow_infinity=False
        )
    )
    y = function(
        hypothesis.strategies.floats(
            min_value=-limit, max_value=limit, allow_nan=False, allow_infinity=False
        )
    )

    input = torch.complex(torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype))

    return input, scipy.special.wofz(input)


@hypothesis.given(_strategy())
def test_faddeeva_w(data):
    input, output = data

    if input.dtype == torch.complex64:
        rtol, atol = 1e-5, 1e-5
    elif input.dtype == torch.complex128:
        rtol, atol = 1e-10, 1e-10
    else:
        rtol, atol = None, None

    torch.testing.assert_close(
        beignet.special.faddeeva_w(input), output, rtol=rtol, atol=atol
    )


def test_faddeeva_w_propagates_nan():
    input = torch.complex(torch.tensor(torch.nan), torch.tensor(torch.nan))
    output = beignet.special.faddeeva_w(input)
    torch.testing.assert_close(input, output, equal_nan=True)
