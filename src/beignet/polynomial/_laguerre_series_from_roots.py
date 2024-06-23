from .__from_roots import _from_roots
from ._laguerre_series_line import laguerre_series_line
from ._multiply_laguerre_series import multiply_laguerre_series


def laguerre_series_from_roots(input):
    return _from_roots(
        laguerre_series_line,
        multiply_laguerre_series,
        input,
    )
