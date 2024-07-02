from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._legvander import legvander


def legvander2d(x: Tensor, y: Tensor, degree: Tensor) -> Tensor:
    return _flattened_vandermonde(
        (legvander, legvander),
        (x, y),
        degree,
    )
