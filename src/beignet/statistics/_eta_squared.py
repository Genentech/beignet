import torch
from torch import Tensor


def eta_squared(
    ss_between: Tensor,
    ss_total: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute eta-squared (η²), a measure of effect size for ANOVA.

    Eta-squared represents the proportion of total variance that is attributed
    to the treatment effect. It is analogous to R² in regression analysis.

    When to Use
    -----------
    **Traditional Statistics:**
    - Reporting effect sizes in ANOVA results
    - Meta-analyses requiring standardized effect measures
    - Comparing relative importance of factors in factorial designs
    - Educational research assessing intervention effectiveness

    **Machine Learning Contexts:**
    - Feature importance assessment in regression models
    - Model comparison: variance explained by different feature sets
    - Hyperparameter analysis: variance explained by different configurations
    - A/B/C testing: comparing effect sizes across multiple treatments
    - Ensemble methods: assessing individual model contributions
    - Cross-validation: variance explained across different folds
    - Domain adaptation: measuring variance explained by domain factors
    - Representation learning: variance captured by different embedding dimensions

    **Choose eta-squared when:**
    - Multiple group comparisons (3+ groups)
    - Need standardized effect size measure (0-1 scale)
    - Comparing effects across different studies or contexts
    - Want to quantify practical significance beyond statistical significance

    **Interpretation (Cohen, 1988):**
    - η² = 0.01: small effect (1% of variance explained)
    - η² = 0.06: medium effect (6% of variance explained)
    - η² = 0.14: large effect (14% of variance explained)

    Parameters
    ----------
    ss_between : Tensor
        Between-groups sum of squares (treatment effect).
    ss_total : Tensor
        Total sum of squares (total variance).

    Returns
    -------
    output : Tensor
        Eta-squared values in range [0, 1].

    Examples
    --------
    >>> ss_between = torch.tensor(150.0)
    >>> ss_total = torch.tensor(500.0)
    >>> eta_squared(ss_between, ss_total)
    tensor(0.3000)

    Notes
    -----
    Eta-squared is defined as:

    η² = SS_between / SS_total

    where:
    - SS_between = sum of squares between groups
    - SS_total = total sum of squares

    Interpretation guidelines (Cohen, 1988):
    - η² = 0.01: small effect
    - η² = 0.06: medium effect
    - η² = 0.14: large effect

    Note: Eta-squared can be biased upward, especially with small samples.
    Consider using partial eta-squared or omega-squared for less biased estimates.

    For ANOVA results from raw data, use:
    SS_total = sum((x - grand_mean)²) for all observations
    SS_between = sum(n_i * (group_mean_i - grand_mean)²) for all groups
    """
    ss_between = torch.atleast_1d(torch.as_tensor(ss_between))
    ss_total = torch.atleast_1d(torch.as_tensor(ss_total))

    # Ensure common floating point dtype
    if ss_between.dtype.is_floating_point and ss_total.dtype.is_floating_point:
        if ss_between.dtype == torch.float64 or ss_total.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    ss_between = ss_between.to(dtype)
    ss_total = ss_total.to(dtype)

    # Validate inputs
    ss_between = torch.clamp(ss_between, min=0.0)
    ss_total = torch.clamp(ss_total, min=torch.finfo(dtype).eps)

    # Ensure ss_between <= ss_total (mathematical constraint)
    ss_between = torch.clamp(ss_between, max=ss_total)

    # Compute eta-squared
    eta_sq = ss_between / ss_total

    # Clamp to valid range [0, 1]
    eta_sq = torch.clamp(eta_sq, 0.0, 1.0)

    if out is not None:
        out.copy_(eta_sq)
        return out
    return eta_sq


def partial_eta_squared(
    ss_effect: Tensor,
    ss_error: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute partial eta-squared (η²p), a measure of effect size for ANOVA.

    Partial eta-squared represents the proportion of variance in the dependent
    variable that is attributed to the treatment effect, after removing variance
    due to other factors. It is less biased than eta-squared and preferred
    for factorial designs.

    Parameters
    ----------
    ss_effect : Tensor
        Sum of squares for the specific effect being tested.
    ss_error : Tensor
        Error sum of squares (within-groups variance).

    Returns
    -------
    output : Tensor
        Partial eta-squared values in range [0, 1].

    Examples
    --------
    >>> ss_effect = torch.tensor(120.0)
    >>> ss_error = torch.tensor(280.0)
    >>> partial_eta_squared(ss_effect, ss_error)
    tensor(0.3000)

    Notes
    -----
    Partial eta-squared is defined as:

    η²p = SS_effect / (SS_effect + SS_error)

    Interpretation guidelines (same as eta-squared):
    - η²p = 0.01: small effect
    - η²p = 0.06: medium effect
    - η²p = 0.14: large effect

    Partial eta-squared is preferred over eta-squared because:
    1. It provides a less biased estimate of effect size
    2. It's more appropriate for factorial designs
    3. It's commonly reported in modern statistical software
    """
    ss_effect = torch.atleast_1d(torch.as_tensor(ss_effect))
    ss_error = torch.atleast_1d(torch.as_tensor(ss_error))

    # Ensure common floating point dtype
    dtype = (
        torch.float64
        if (ss_effect.dtype == torch.float64 or ss_error.dtype == torch.float64)
        else torch.float32
    )
    ss_effect = ss_effect.to(dtype)
    ss_error = ss_error.to(dtype)

    # Validate inputs
    ss_effect = torch.clamp(ss_effect, min=0.0)
    ss_error = torch.clamp(ss_error, min=torch.finfo(dtype).eps)

    # Compute partial eta-squared
    partial_eta_sq = ss_effect / (ss_effect + ss_error)

    # Clamp to valid range [0, 1]
    partial_eta_sq = torch.clamp(partial_eta_sq, 0.0, 1.0)

    if out is not None:
        out.copy_(partial_eta_sq)
        return out
    return partial_eta_sq
