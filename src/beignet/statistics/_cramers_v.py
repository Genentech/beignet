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

    When to Use
    -----------
    **Traditional Statistics:**
    - Measuring association strength between categorical variables
    - Reporting effect sizes for chi-square independence tests
    - Social science research with nominal outcomes
    - Market research analyzing consumer behavior categories

    **Machine Learning Contexts:**
    - Feature selection: measuring association between categorical features
    - Model evaluation: association between predicted and true categories
    - Fairness assessment: measuring association between outcomes and demographics
    - Data quality: detecting unwanted associations in categorical data
    - A/B testing: measuring association between treatments and categorical outcomes
    - Natural language processing: association between text categories and outcomes
    - Recommendation systems: user-item category associations
    - Computer vision: association between image categories and labels
    - Customer segmentation: association strength between segment variables

    **Choose Cramer's V when:**
    - Both variables are categorical (nominal or ordinal)
    - Contingency tables larger than 2×2 (use phi coefficient for 2×2)
    - Need standardized association measure (0-1 scale)
    - Comparing association strength across different table sizes

    **Interpretation Guidelines:**
    - V = 0: no association
    - V = 0.1: weak association
    - V = 0.3: moderate association
    - V = 0.5: strong association
    - V = 1.0: perfect association

    Parameters
    ----------
    chi_square : Tensor
        Chi-square test statistic. Can be a scalar or tensor.

    sample_size : Tensor
        Total sample size (sum of all cells in the contingency table).

    min_dim : Tensor
        Minimum of (number of rows - 1, number of columns - 1).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Cramer's V effect size (between 0 and 1).

    Examples
    --------
    >>> chi_sq = torch.tensor(10.5)
    >>> n = torch.tensor(100)
    >>> min_dim = torch.tensor(1)
    >>> cramers_v(chi_sq, n, min_dim)
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
    chi_square = torch.atleast_1d(torch.as_tensor(chi_square))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    min_dim = torch.atleast_1d(torch.as_tensor(min_dim))

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

    output = torch.sqrt(chi_square / (sample_size * min_dim))

    output = torch.clamp(output, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
