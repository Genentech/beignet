from .__from_roots import _from_roots
from ._chebyshev_series_line import chebyshev_series_line
from ._multiply_chebyshev_series import multiply_chebyshev_series


def chebfromroots(input):
    return _from_roots(chebyshev_series_line, multiply_chebyshev_series, input)
