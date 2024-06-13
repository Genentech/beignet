from beignet._multiply_probabilists_hermite_series import (
    multiply_probabilists_hermite_series,
)

from .polynomial.__fromroots import _fromroots
from .polynomial._hermeline import hermeline


def probabilists_hermite_series_from_roots(roots):
    return _fromroots(hermeline, multiply_probabilists_hermite_series, roots)
