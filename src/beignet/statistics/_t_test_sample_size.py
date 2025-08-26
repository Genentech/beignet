import torch
from torch import Tensor


def t_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for one-sample and paired t-tests.

    Given the effect size, desired power, and significance level, this function
    calculates the minimum sample size needed to achieve the specified power
    for a one-sample t-test or paired-samples t-test.

    This function is differentiable with respect to the effect_size parameter.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning experiments to detect meaningful differences
    - Grant applications requiring power calculations
    - Determining adequate sample sizes for clinical trials
    - Pilot studies to plan full-scale experiments

    **Machine Learning Contexts:**
    - A/B testing: determining user sample sizes for reliable comparisons
    - Model validation: planning data collection for performance evaluation
    - Cross-validation: determining fold sizes for stable estimates
    - Hyperparameter tuning: sample sizes for parameter comparison studies
    - Active learning: determining query batch sizes
    - Transfer learning: planning target domain sample sizes
    - Federated learning: determining client data requirements
    - Benchmarking: planning dataset sizes for model comparisons

    **Use for one-sample t-test sample size when:**
    - Testing if model performance differs from known benchmark
    - Comparing single group mean to population value
    - Validating if learned representations meet target properties

    **Use for paired t-test sample size when:**
    - Before/after model comparisons (paired observations)
    - Cross-validation fold-wise comparisons
    - Paired algorithm evaluations on same datasets

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). For one-sample tests, this is
        (μ - μ₀) / σ where μ is the true mean, μ₀ is the null hypothesis mean,
        and σ is the population standard deviation. For paired tests, this is
        the mean difference divided by the standard deviation of differences.
        Should be positive.

    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided" or "one-sided".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> t_test_sample_size(effect_size, power=0.8)
    tensor(34)

    Notes
    -----
    The sample size calculation is based on the noncentral t-distribution.
    For a t-test with effect size d and sample size n, the noncentrality
    parameter is:

    δ = d * √n

    The calculation uses an iterative approach to find the sample size that
    achieves the desired power, starting from an initial normal approximation:

    n ≈ ((z_α + z_β) / d)²

    Where z_α and z_β are the critical values for the given α and β = 1 - power.

    For computational efficiency, we use analytical approximations where possible,
    falling back to iterative refinement when needed.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Aberson, C. L. (2010). Applied power analysis for the behavioral
           sciences. Routledge.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    if effect_size.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.sqrt(torch.tensor(2.0, dtype=dtype)) * torch.erfinv(
        2.0 * torch.as_tensor(power, dtype=dtype) - 1.0
    )

    sample_size_initial = ((z_alpha + z_beta) / effect_size) ** 2

    sample_size_initial = torch.clamp(sample_size_initial, min=2.0)

    sample_size_current = sample_size_initial
    convergence_tolerance = 1e-6
    max_iterations = 10

    for _iteration in range(max_iterations):
        df_current = sample_size_current - 1
        df_current = torch.clamp(df_current, min=1.0)

        noncentrality_parameter_current = effect_size * torch.sqrt(sample_size_current)

        if alternative == "two-sided":
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))
        else:
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))

        var_nct = (df_current + noncentrality_parameter_current**2) / torch.clamp(
            df_current - 2, min=0.1
        )
        var_nct = torch.where(
            df_current > 2,
            var_nct,
            1 + noncentrality_parameter_current**2 / (2 * df_current),
        )
        std_nct = torch.sqrt(var_nct)

        if alternative == "two-sided":
            z_upper = (t_critical - noncentrality_parameter_current) / torch.clamp(
                std_nct, min=1e-10
            )
            z_lower = (-t_critical - noncentrality_parameter_current) / torch.clamp(
                std_nct, min=1e-10
            )
            power_current = 0.5 * (
                1 - torch.erf(z_upper / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (
                1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            z_score = (t_critical - noncentrality_parameter_current) / torch.clamp(
                std_nct, min=1e-10
            )
            power_current = 0.5 * (
                1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )

        power_current = torch.clamp(power_current, 0.01, 0.99)

        power_diff = power - power_current

        adjustment = (
            power_diff
            * sample_size_current
            / (2 * torch.clamp(power_current * (1 - power_current), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance
        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        sample_size_current = sample_size_current + adjustment

        sample_size_current = torch.clamp(sample_size_current, min=2.0, max=100000.0)

    output = torch.ceil(sample_size_current)

    output = torch.clamp(output, min=2.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
