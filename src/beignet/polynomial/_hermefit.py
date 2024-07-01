from torch import Tensor

from beignet.polynomial import hermevander
from beignet.polynomial.__fit import _fit


def hermefit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
):
    return _fit(
        hermevander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )
