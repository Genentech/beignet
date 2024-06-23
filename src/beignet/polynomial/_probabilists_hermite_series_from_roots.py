from torch import Tensor

from .__from_roots import _from_roots
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series
from ._probabilists_hermite_series_line import probabilists_hermite_series_line


def probabilists_hermite_series_from_roots(input: Tensor) -> Tensor:
    return _from_roots(
        probabilists_hermite_series_line,
        multiply_probabilists_hermite_series,
        input,
    )
