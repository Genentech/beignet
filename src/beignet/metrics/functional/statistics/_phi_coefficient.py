"""Phi coefficient functional metric."""

import torch
from torch import Tensor

import beignet.statistics


def phi_coefficient(
    contingency_table: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute the phi coefficient for a 2x2 contingency table.

    Parameters
    ----------
    contingency_table : Tensor, shape=(..., 2, 2)
        A 2x2 contingency table.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        The phi coefficient values.
    """
    # Ensure the contingency table is at least 2D
    contingency_table = torch.atleast_2d(contingency_table)

    # Compute chi-square statistic from contingency table
    # For 2x2 table: chi_square = N * (ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d))
    a = contingency_table[..., 0, 0]
    b = contingency_table[..., 0, 1]
    c = contingency_table[..., 1, 0]
    d = contingency_table[..., 1, 1]

    sample_size = contingency_table.sum(dim=(-2, -1))

    numerator = sample_size * (a * d - b * c) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)

    # Handle division by zero
    denominator = torch.clamp(denominator, min=torch.finfo(contingency_table.dtype).eps)
    chi_square = numerator / denominator

    # Use the statistics function to compute phi coefficient
    result = beignet.statistics.phi_coefficient(chi_square, sample_size)

    if out is not None:
        out.copy_(result)
        return out

    return result
