import torch
from torch import Tensor


def cramers_v(
    chi_square: Tensor,
    sample_size: Tensor,
    min_dim: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Cramer's V effect size for chi-square tests.

    Cramer's V is a measure of association between two nominal variables,
    giving a value between 0 and 1 (inclusive). It's based on the chi-square
    statistic and can be used with contingency tables of any size.

    This function is differentiable with respect to all input parameters.
    While differentiability isn't typically needed for effect size calculations,
    it enables integration into machine learning pipelines where chi-square
    statistics may be computed from learned representations.

    Parameters
    ----------
    chi_square : Tensor
        Chi-square test statistic. Can be a scalar or tensor.
    sample_size : Tensor
        Total sample size (sum of all cells in the contingency table).
    min_dim : Tensor
        Minimum of (number of rows - 1, number of columns - 1).

    Returns
    -------
    output : Tensor
        Cramer's V effect size (between 0 and 1).

    Examples
    --------
    >>> chi_sq = torch.tensor(10.5)
    >>> n = torch.tensor(100)
    >>> min_dim = torch.tensor(1)  # 2x2 table: min(2-1, 2-1) = 1
    >>> beignet.cramers_v(chi_sq, n, min_dim)
    tensor(0.3240)

    Notes
    -----
    For a 2×2 contingency table, Cramer's V is equivalent to the absolute
    value of the phi coefficient. For larger tables, it provides a normalized
    measure of association that accounts for the table size.

    The interpretation guidelines (Cohen, 1988):
    - Small effect: V = 0.10
    - Medium effect: V = 0.30
    - Large effect: V = 0.50
    """
    # Convert inputs to tensors if needed
    chi_square = torch.as_tensor(chi_square)
    sample_size = torch.as_tensor(sample_size)
    min_dim = torch.as_tensor(min_dim)

    # Ensure all have the same dtype
    dtype = chi_square.dtype
    if sample_size.dtype != dtype:
        if sample_size.dtype == torch.float64 or dtype == torch.float64:
            dtype = torch.float64
    if min_dim.dtype != dtype:
        if min_dim.dtype == torch.float64 or dtype == torch.float64:
            dtype = torch.float64

    chi_square = chi_square.to(dtype)
    sample_size = sample_size.to(dtype)
    min_dim = min_dim.to(dtype)

    # Cramer's V formula: V = sqrt(χ² / (n * min_dim))
    # where min_dim = min(r-1, c-1) for r rows and c columns
    output = torch.sqrt(chi_square / (sample_size * min_dim))

    # Clamp to [0, 1] range to handle numerical errors
    output = torch.clamp(output, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
