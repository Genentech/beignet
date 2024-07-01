from torch import Tensor

from beignet.polynomial import _fit, hermvander


def hermfit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
):
    return _fit(
        hermvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )
