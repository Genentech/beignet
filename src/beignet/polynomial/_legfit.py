from torch import Tensor

from .__fit import _fit
from ._legvander import legvander


def legfit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
):
    return _fit(
        legvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )
