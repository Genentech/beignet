from torch import Tensor

from beignet.polynomial import lagvander
from beignet.polynomial.__fit import _fit


def lagfit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
):
    return _fit(
        lagvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )
