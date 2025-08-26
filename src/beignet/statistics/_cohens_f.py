import torch
from torch import Tensor


def cohens_f(
    group_means: Tensor,
    pooled_std: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Cohen's f effect size for ANOVA.

    Cohen's f is a measure of effect size for ANOVA that quantifies the
    standardized variability between group means. It is defined as the
    standard deviation of the group means divided by the pooled within-group
    standard deviation.

    This function is differentiable with respect to both input parameters.
    While effect size calculations don't typically require gradients,
    differentiability enables integration into machine learning pipelines
    where group statistics might be computed from learned representations.

    When to Use
    -----------
    **Traditional Statistics:**
    - Reporting standardized effect sizes in ANOVA results
    - Meta-analyses requiring comparable effect measures across studies
    - Planning studies with multiple group comparisons
    - Converting between different effect size measures (f = d/2 for two groups)

    **Machine Learning Contexts:**
    - Multi-model performance comparisons (3+ models)
    - Hyperparameter optimization: comparing effect sizes across configurations
    - Feature selection: comparing importance across multiple feature sets
    - A/B/C+ testing: standardized effect sizes for multi-variant experiments
    - Ensemble model evaluation: measuring relative contributions
    - Cross-validation: effect sizes across multiple validation folds
    - Multi-domain learning: comparing performance across domains
    - Clustering evaluation: comparing separation across different algorithms

    **Choose Cohen's f when:**
    - Comparing 3+ groups (use Cohen's d for 2 groups)
    - Need ANOVA-specific effect size measure
    - Converting to/from other effect size measures
    - Planning sample sizes for multi-group studies

    **Interpretation Guidelines:**
    - f = 0.1: small effect
    - f = 0.25: medium effect
    - f = 0.4: large effect

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
        Cohen's f effect size.

    Examples
    --------
    >>> group_means = torch.tensor([10.0, 12.0, 14.0])
    >>> pooled_std = torch.tensor(2.0)
    >>> cohens_f(group_means, pooled_std)
    tensor(1.0000)

    Notes
    -----
    Cohen's f is calculated as:

    f = σ_means / σ_pooled

    Where σ_means is the standard deviation of the group means and σ_pooled
    is the pooled within-group standard deviation.

    The relationship to other effect sizes:
    - f² = η² / (1 - η²), where η² is eta-squared
    - For two groups: f = d / 2, where d is Cohen's d

    Interpretation guidelines (Cohen, 1988):
    - Small effect: f = 0.10
    - Medium effect: f = 0.25
    - Large effect: f = 0.40

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Lakens, D. (2013). Calculating and reporting effect sizes to
           facilitate cumulative science. Frontiers in Psychology, 4, 863.
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

    # Cohen's f = σ_means / σ_pooled
    output = sigma_means / pooled_std_safe

    if out is not None:
        out.copy_(output)
        return out

    return output
