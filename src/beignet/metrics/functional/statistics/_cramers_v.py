"""Cramer's V effect size functional metric."""

import torch
from torch import Tensor

import beignet.statistics


def cramers_v(
    contingency_table: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute Cramer's V effect size from a contingency table.

    Parameters
    ----------
    contingency_table : Tensor, shape=(..., rows, cols)
        A contingency table.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        The Cramer's V values.
    """
    # Ensure the contingency table is at least 2D
    contingency_table = torch.atleast_2d(contingency_table)

    # Compute expected frequencies under independence
    row_totals = contingency_table.sum(dim=-1, keepdim=True)
    col_totals = contingency_table.sum(dim=-2, keepdim=True)
    sample_size = contingency_table.sum(dim=(-2, -1), keepdim=True)

    expected = (row_totals * col_totals) / sample_size

    # Compute chi-square statistic
    chi_square = torch.sum(
        (contingency_table - expected) ** 2
        / torch.clamp(expected, min=torch.finfo(contingency_table.dtype).eps),
        dim=(-2, -1),
    )

    # Get sample size (remove keepdim for final calculation)
    sample_size = sample_size.squeeze((-2, -1))

    # Get minimum dimension minus 1 for Cramer's V
    min_dim = torch.tensor(
        min(contingency_table.shape[-2], contingency_table.shape[-1]) - 1,
        dtype=contingency_table.dtype,
        device=contingency_table.device,
    )
    min_dim = min_dim.expand(chi_square.shape)

    # Use the statistics function to compute Cramer's V
    result = beignet.statistics.cramers_v(chi_square, sample_size, min_dim)

    if out is not None:
        out.copy_(result)
        return out

    return result
