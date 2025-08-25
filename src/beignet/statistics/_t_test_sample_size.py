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
    # Convert inputs to tensors if needed
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    # Ensure effect_size has appropriate dtype
    if effect_size.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    # Clamp effect size to positive values
    effect_size = torch.clamp(effect_size, min=1e-6)

    # Standard normal quantiles using erfinv
    # Normalize alternative
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Use same historical approximation as power function for consistency
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    # z_beta = Phi^{-1}(power)
    z_beta = torch.sqrt(torch.tensor(2.0, dtype=dtype)) * torch.erfinv(
        2.0 * torch.as_tensor(power, dtype=dtype) - 1.0
    )

    # Initial normal approximation: n = ((z_alpha + z_beta) / d)^2
    n_initial = ((z_alpha + z_beta) / effect_size) ** 2

    # Ensure minimum sample size
    n_initial = torch.clamp(n_initial, min=2.0)

    # Iterative refinement to account for finite df effects
    n_current = n_initial
    convergence_tolerance = 1e-6
    max_iterations = 10

    for _iteration in range(max_iterations):
        # Calculate current degrees of freedom
        df_current = n_current - 1
        df_current = torch.clamp(df_current, min=1.0)

        # Current noncentrality parameter
        ncp_current = effect_size * torch.sqrt(n_current)

        # Adjust critical value for finite df
        if alternative == "two-sided":
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))
        else:
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))

        # Variance of noncentral t-distribution
        var_nct = (df_current + ncp_current**2) / torch.clamp(df_current - 2, min=0.1)
        # Use more stable approximation for small df
        var_nct = torch.where(
            df_current > 2, var_nct, 1 + ncp_current**2 / (2 * df_current)
        )
        std_nct = torch.sqrt(var_nct)

        # Calculate current power
        if alternative == "two-sided":
            z_upper = (t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)
            z_lower = (-t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)
            power_current = 0.5 * (
                1 - torch.erf(z_upper / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (
                1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            z_score = (t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)
            power_current = 0.5 * (
                1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )

        # Clamp power to valid range
        power_current = torch.clamp(power_current, 0.01, 0.99)

        # Calculate power difference
        power_diff = power - power_current

        # Newton-Raphson style adjustment with convergence damping
        # Approximate derivative: d(power)/d(n) ≈ d(power)/d(ncp) * d(ncp)/d(n)
        # d(ncp)/d(n) = effect_size / (2 * sqrt(n))
        # Adjust sample size based on power difference
        adjustment = (
            power_diff
            * n_current
            / (2 * torch.clamp(power_current * (1 - power_current), min=0.01))
        )

        # Dampen adjustment if close to convergence (compile-friendly)
        converged_mask = torch.abs(power_diff) < convergence_tolerance
        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        n_current = n_current + adjustment

        # Ensure minimum constraints
        n_current = torch.clamp(n_current, min=2.0, max=100000.0)

    # Round up to nearest integer
    output = torch.ceil(n_current)

    # Final check: ensure we have at least 2 subjects
    output = torch.clamp(output, min=2.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
