import torch
from torch import Tensor


def cohens_d(
    group1: Tensor,
    group2: Tensor,
    pooled: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Cohen's d effect size between two groups.

    Cohen's d is a standardized effect size measure that quantifies the difference
    between two groups in terms of their pooled standard deviation.

    This function is fully differentiable with respect to both group inputs.
    While differentiability is not typically needed for traditional statistical
    analysis, it enables integration into machine learning pipelines where effect
    sizes may be computed from learned representations or used in gradient-based
    hyperparameter optimization.

    Parameters
    ----------
    group1 : Tensor, shape (..., N1)
        First group of observations. The last dimension contains the observations.

    group2 : Tensor, shape (..., N2)
        Second group of observations. The last dimension contains the observations.

    pooled : bool, default=True
        If True, use pooled standard deviation. If False, use the standard
        deviation of group1 as the denominator.

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor, shape (...)
        Cohen's d effect size. Positive values indicate group1 > group2.

    Examples
    --------
    >>> group1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> group2 = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    >>> cohens_d(group1, group2)
    tensor(-0.7071)
    """
    mean1 = torch.mean(group1, dim=-1)
    mean2 = torch.mean(group2, dim=-1)

    if pooled:
        # Pooled standard deviation
        n1 = group1.shape[-1]
        n2 = group2.shape[-1]

        var1 = torch.var(group1, dim=-1, unbiased=True)
        var2 = torch.var(group2, dim=-1, unbiased=True)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = torch.sqrt(pooled_var)

        output = (mean1 - mean2) / pooled_std
    else:
        # Use group1 standard deviation
        std1 = torch.std(group1, dim=-1, unbiased=True)
        output = (mean1 - mean2) / std1

    if out is not None:
        out.copy_(output)
        return out

    return output
