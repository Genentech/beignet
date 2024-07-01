from torch import Tensor

from beignet.polynomial import polyvander
from beignet.polynomial.__fit import _fit


def polyfit(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
):
    r"""
    Parameters
    ----------
    input : Tensor
        Independent variable.

    other : Tensor
        Dependent variable.

    degree : Tensor or int
        Degree of the fitting polynomial.

    relative_condition : float, optional
        Relative condition number.

    full : bool, default=False
        Return additional information.

    weight : Tensor, optional
        Weights.

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the fit.
    """
    return _fit(
        polyvander,
        input,
        other,
        degree,
        relative_condition,
        full,
        weight,
    )
