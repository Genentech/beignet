import torch
from torch import Tensor


def cohens_f_squared(
    group_means: Tensor,
    pooled_std: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Cohen's f² effect size for ANOVA.

    Cohen's f² is the squared version of Cohen's f effect size, providing
    a measure of the proportion of variance explained by group differences.
    It is related to eta-squared (η²) and partial eta-squared.

    This function is differentiable with respect to both input parameters.
    While effect size calculations don't typically require gradients,
    differentiability enables integration into machine learning pipelines
    where group statistics might be computed from learned representations.

    Parameters
    ----------
    group_means : Tensor, shape (..., k)
        Means for each group. The last dimension represents different groups.

    pooled_std : Tensor
        Pooled within-group standard deviation. Should be broadcastable with group_means.

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Cohen's f² effect size.

    Examples
    --------
    >>> group_means = torch.tensor([10.0, 12.0, 14.0])
    >>> pooled_std = torch.tensor(2.0)
    >>> cohens_f_squared(group_means, pooled_std)
    tensor(1.0000)

    Notes
    -----
    Cohen's f² is calculated as:

    f² = (σ_means / σ_pooled)²

    Where σ_means is the standard deviation of the group means and σ_pooled
    is the pooled within-group standard deviation.

    The relationship to other effect sizes:
    - f² = η² / (1 - η²), where η² is eta-squared
    - η² = f² / (1 + f²)
    - For two groups: f² = d² / 4, where d is Cohen's d

    Interpretation guidelines (Cohen, 1988):
    - Small effect: f² = 0.01 (η² ≈ 0.01)
    - Medium effect: f² = 0.0625 (η² ≈ 0.059)
    - Large effect: f² = 0.16 (η² ≈ 0.138)

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Richardson, J. T. E. (2011). Eta squared and partial eta squared
           as measures of effect size in educational research. Educational
           Research Review, 6(2), 135-147.
    """
    # Convert inputs to tensors if needed
    group_means = torch.atleast_1d(torch.as_tensor(group_means))
    pooled_std = torch.atleast_1d(torch.as_tensor(pooled_std))

    # Ensure both tensors have the same dtype
    if group_means.dtype == torch.float64 or pooled_std.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    group_means = group_means.to(dtype)
    pooled_std = pooled_std.to(dtype)

    # Calculate the standard deviation of group means
    # If group_means has shape (..., k), compute std along last dimension
    sigma_means = torch.std(group_means, dim=-1, unbiased=False)

    # Avoid division by zero
    pooled_std_safe = torch.where(
        torch.abs(pooled_std) < 1e-10,
        torch.tensor(1e-10, dtype=dtype, device=pooled_std.device),
        pooled_std,
    )

    # Cohen's f² = (σ_means / σ_pooled)²
    cohens_f_value = sigma_means / pooled_std_safe
    output = cohens_f_value**2

    if out is not None:
        out.copy_(output)
        return out

    return output
