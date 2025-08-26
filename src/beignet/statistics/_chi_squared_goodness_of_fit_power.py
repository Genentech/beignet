"""Chi-square goodness-of-fit power."""

import torch
from torch import Tensor


def chi_square_goodness_of_fit_power(
    effect_size: Tensor,
    sample_size: Tensor,
    df: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for chi-square goodness-of-fit tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a chi-square
    goodness-of-fit test.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Testing if observed categorical data follows an expected distribution
    - Validating theoretical models against empirical data
    - Quality control in manufacturing (defect rate distributions)
    - Genetics: testing Hardy-Weinberg equilibrium

    **Machine Learning Contexts:**
    - Validating synthetic data generators (categorical feature distributions)
    - Testing if model predictions follow expected class distributions
    - Fairness auditing: checking if algorithmic decisions match demographic distributions
    - Feature engineering validation for categorical transformations
    - A/B testing with categorical outcomes (user behavior patterns)
    - Natural language processing: validating text generation models
    - Recommendation systems: testing if item popularity follows expected distributions
    - Computer vision: validating class balance in generated datasets

    **Choose chi-square goodness-of-fit over other tests when:**
    - Outcome variable is categorical (nominal or ordinal)
    - Expected cell frequencies are ≥ 5 in all categories
    - Testing against a specific theoretical distribution
    - Need to compare observed vs. expected frequencies across multiple categories

    **Common Effect Sizes (Cohen's w):**
    - w = 0.1: small effect
    - w = 0.3: medium effect
    - w = 0.5: large effect

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. This is calculated as the square root of the
        sum of squared standardized differences: w = √(Σ((p₁ᵢ - p₀ᵢ)²/p₀ᵢ))
        where p₀ᵢ are the expected proportions and p₁ᵢ are the observed proportions.
        Should be non-negative.

    sample_size : Tensor
        Sample size (total number of observations).

    df : Tensor
        Degrees of freedom for the chi-square test (number of categories - 1).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> sample_size = torch.tensor(100)
    >>> df = torch.tensor(3)
    >>> chi_square_goodness_of_fit_power(effect_size, sample_size, df)
    tensor(0.6740)

    Notes
    -----
    The power calculation uses the noncentral chi-square distribution. Under the
    null hypothesis, the test statistic follows χ²(df). Under the alternative
    hypothesis, it follows a noncentral chi-square distribution with noncentrality
    parameter:

    λ = n * w²

    Where n is the sample size and w is Cohen's w effect size.

    Cohen's w effect size interpretation:
    - Small effect: w = 0.10
    - Medium effect: w = 0.30
    - Large effect: w = 0.50

    For computational efficiency, we use normal approximations for large degrees
    of freedom and accurate noncentral chi-square calculations for smaller df.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cramér, H. (1946). Mathematical Methods of Statistics. Princeton
           University Press.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    degrees_of_freedom = torch.atleast_1d(torch.as_tensor(df))

    if (
        effect_size.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or degrees_of_freedom.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    degrees_of_freedom = degrees_of_freedom.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    sample_size = torch.clamp(sample_size, min=1.0)
    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    noncentrality_parameter = sample_size * effect_size**2

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * torch.sqrt(
        torch.tensor(2.0, dtype=dtype)
    )
    chi2_critical = degrees_of_freedom + z_alpha * torch.sqrt(2 * degrees_of_freedom)

    mean_nc_chi2 = degrees_of_freedom + noncentrality_parameter
    var_nc_chi2 = 2 * (degrees_of_freedom + 2 * noncentrality_parameter)
    std_nc_chi2 = torch.sqrt(var_nc_chi2)

    z_score = (chi2_critical - mean_nc_chi2) / torch.clamp(std_nc_chi2, min=1e-10)

    power = 0.5 * (1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=dtype))))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
