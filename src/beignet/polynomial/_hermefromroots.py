from .__from_roots import _from_roots
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series
from ._probabilists_hermite_series_line import probabilists_hermite_series_line


def hermefromroots(input):
    return _from_roots(
        probabilists_hermite_series_line,
        multiply_probabilists_hermite_series,
        input,
    )
