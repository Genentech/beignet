import torch
from torch import Tensor


def phi_coefficient(
    chi_square: Tensor,
    sample_size: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute phi coefficient effect size for 2×2 chi-square tests.

    The phi coefficient is a measure of association for two binary variables,
    equivalent to the Pearson correlation coefficient when both variables are
    binary. It ranges from -1 to +1, where 0 indicates no association.

    This function is differentiable with respect to both input parameters.
    While differentiability isn't typically needed for effect size calculations,
    it enables integration into machine learning pipelines where chi-square
    statistics may be computed from learned representations.

    Parameters
    ----------
    chi_square : Tensor
        Chi-square test statistic from a 2×2 contingency table.

    sample_size : Tensor
        Total sample size (sum of all cells in the contingency table).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Phi coefficient (between -1 and 1).

    Examples
    --------
    >>> chi_sq = torch.tensor(6.25)
    >>> n = torch.tensor(100)
    >>> beignet.phi_coefficient(chi_sq, n)
    tensor(0.2500)

    Notes
    -----
    The phi coefficient is only appropriate for 2×2 contingency tables.
    For larger tables, use Cramer's V instead.

    The interpretation guidelines (Cohen, 1988):
    - Small effect: |φ| = 0.10
    - Medium effect: |φ| = 0.30
    - Large effect: |φ| = 0.50

    The sign of phi depends on the arrangement of the data in the contingency
    table and indicates the direction of association.
    """
    # Convert inputs to tensors if needed
    chi_square = torch.atleast_1d(torch.as_tensor(chi_square))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    # Ensure both have the same dtype
    if chi_square.dtype != sample_size.dtype:
        if chi_square.dtype == torch.float64 or sample_size.dtype == torch.float64:
            chi_square = chi_square.to(torch.float64)
            sample_size = sample_size.to(torch.float64)
        else:
            chi_square = chi_square.to(torch.float32)
            sample_size = sample_size.to(torch.float32)

    # Phi coefficient formula: φ = sqrt(χ² / n)
    # Note: This gives the absolute value of phi. The sign would need
    # to be determined from the original contingency table structure.
    output = torch.sqrt(chi_square / sample_size)

    # Clamp to [0, 1] range (absolute value)
    output = torch.clamp(output, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
