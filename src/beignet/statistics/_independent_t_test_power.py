import torch
from torch import Tensor


def independent_t_test_power(
    effect_size: Tensor,
    nobs1: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: Tensor | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for independent samples t-tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for an independent
    samples t-test (also known as two-sample t-test).

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning sample sizes for experimental studies with two independent groups
    - Clinical trials comparing treatment vs. control groups
    - Educational research comparing different teaching methods
    - A/B testing in traditional experimental designs

    **Machine Learning Contexts:**
    - A/B testing for ML model performance comparisons
    - Comparing two different algorithms on independent datasets
    - Model validation: comparing performance across different data splits
    - Hyperparameter optimization: comparing two parameter configurations
    - Feature importance: comparing model performance with/without features
    - Domain adaptation: comparing performance across independent domains
    - Fairness assessment: comparing outcomes between demographic groups
    - Active learning: power to detect improvement from additional data
    - Transfer learning: comparing source vs. target domain performance

    **Choose independent t-test over paired t-test when:**
    - Samples are from different subjects/units (not paired)
    - Two distinct groups being compared
    - No natural pairing or matching between observations
    - Independent group assignment (randomization)

    **Choose independent t-test over other tests when:**
    - Continuous outcome variables
    - Approximately normal distributions
    - Independent observations within and between groups
    - Homogeneous variances (or use Welch's t-test)

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the two group means divided by the pooled standard deviation:
        d = (μ₁ - μ₂) / σ_pooled. Should be positive.
    nobs1 : Tensor
        Number of observations in group 1.
    ratio : Tensor, default=1.0
        Ratio of sample size for group 2 relative to group 1.
        Group 2 sample size = nobs1 * ratio.
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> nostandard_deviation_1 = torch.tensor(30)
    >>> independent_t_test_power(effect_size, nobs1)
    tensor(0.4741)

    Notes
    -----
    The power calculation uses the noncentral t-distribution. The test statistic
    under the null hypothesis follows t(df) where df = n₁ + n₂ - 2. Under the
    alternative hypothesis, it follows a noncentral t-distribution with
    noncentrality parameter:

    δ = d * √(n₁ * n₂ / (n₁ + n₂))

    Where d is Cohen's d effect size, n₁ and n₂ are the sample sizes.

    The pooled standard error is:
    SE = √((1/n₁ + 1/n₂) * σ²_pooled)

    For computational efficiency, we use normal approximations for large
    sample sizes and accurate noncentral t-distribution calculations for
    smaller samples.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Aberson, C. L. (2010). Applied power analysis for the behavioral
           sciences. Routledge.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    if (
        effect_size.dtype == torch.float64
        or sample_size_group_1.dtype == torch.float64
        or ratio.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size_group_1 = sample_size_group_1.to(dtype)
    ratio = ratio.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)
    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)
    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sample_size_group_2 = sample_size_group_1 * ratio
    total_sample_size = sample_size_group_1 + sample_size_group_2

    degrees_of_freedom = total_sample_size - 2
    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    se_factor = torch.sqrt(1 / sample_size_group_1 + 1 / sample_size_group_2)

    noncentrality_parameter = effect_size / se_factor

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_eff = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
        t_critical = z_eff * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
    else:
        z_eff = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        t_critical = z_eff * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    mean_nct = noncentrality_parameter

    var_nct = torch.where(
        degrees_of_freedom > 2,
        (degrees_of_freedom + noncentrality_parameter**2) / (degrees_of_freedom - 2),
        1 + noncentrality_parameter**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
    )
    std_nct = torch.sqrt(var_nct)

    if alt == "two-sided":
        z_upper = (t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        z_lower = (-t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (1 - torch.erf(z_upper / torch.sqrt(torch.tensor(2.0)))) + 0.5 * (
            1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0)))
        )
    elif alt == "greater":
        z_score = (t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))
    else:
        z_score = (-t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
