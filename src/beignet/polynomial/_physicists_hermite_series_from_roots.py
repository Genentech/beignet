from torch import Tensor

from .__from_roots import _from_roots
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series
from ._physicists_hermite_series_line import physicists_hermite_series_line


def physicists_hermite_series_from_roots(input: Tensor) -> Tensor:
    return _from_roots(
        physicists_hermite_series_line,
        multiply_physicists_hermite_series,
        input,
    )
