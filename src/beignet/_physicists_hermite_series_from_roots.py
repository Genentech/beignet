from beignet._multiply_physicists_hermite_series import (
    multiply_physicists_hermite_series,
)

from .polynomial.__fromroots import _fromroots
from .polynomial._hermline import hermline


def physicists_hermite_series_from_roots(roots):
    return _fromroots(hermline, multiply_physicists_hermite_series, roots)
