from .__from_roots import _from_roots
from ._chebline import chebline
from ._multiply_chebyshev_series import multiply_chebyshev_series


def chebfromroots(input):
    return _from_roots(chebline, multiply_chebyshev_series, input)
