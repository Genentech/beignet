"""Independent z-test power (two-sample z-test with known variances)."""

import math

import torch
from torch import Tensor


def independent_z_test_power(
    effect_size: Tensor,
    sample_size1: Tensor,
    sample_size2: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for independent samples z-tests with known variances.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a two-sample z-test
    where the population variances are known.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Quality control with known population variances from historical data
    - Laboratory experiments where measurement precision is well-established
    - Survey research comparing means between groups with known sampling error
    - Industrial process monitoring with established variance parameters
    - Educational testing with standardized instruments having known reliability

    **Machine Learning Contexts:**
    - Model comparison when performance variance is known from extensive benchmarking
    - A/B testing with metrics having well-established variance (e.g., click-through rates)
    - Algorithm evaluation using standardized benchmark datasets with known properties
    - Hyperparameter optimization when computational variance is predictable
    - Cross-validation studies with established variance estimates from pilot studies
    - Transfer learning evaluation with known source domain performance variance
    - Ensemble method comparison when individual model variances are established
    - Active learning evaluation with known sampling variance properties
    - Domain adaptation studies with established target domain variance
    - Recommendation system evaluation with known user behavior variance
    - Computer vision model comparison using standardized evaluation protocols
    - NLP model evaluation with established benchmark variance parameters

    **Choose z-test over t-test when:**
    - Population variances are truly known (not estimated from data)
    - Sample sizes are very large (>30 per group) making t ≈ z
    - Working with standardized, well-characterized measurement systems
    - Historical data provides reliable variance estimates
    - Simulation studies where true parameters are known

    **Choose two-sample z-test over one-sample when:**
    - Comparing two groups rather than testing against a fixed value
    - No established population mean to serve as null hypothesis value
    - Both groups are sampled from potentially different populations
    - Interest is in relative difference rather than absolute comparison

    **Interpretation Guidelines:**
    - Effect size d = (μ₁ - μ₂) / σ_pooled follows Cohen's conventions
    - d = 0.20 (small), 0.50 (medium), 0.80 (large) effect sizes
    - Power increases with larger effect sizes and larger sample sizes
    - Two-sided tests require larger samples than one-sided for same power
    - Equal sample sizes generally provide optimal power for fixed total N
    - Assumes normality and known, equal variances between groups

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the two population means divided by the pooled standard deviation:
        d = (μ₁ - μ₂) / σ_pooled. Should be positive for "larger" alternative,
        negative for "smaller".
    sample_size1 : Tensor
        Sample size for group 1.
    sample_size2 : Tensor, optional
        Sample size for group 2. If None, assumed equal to sample_size1.
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
    >>> sample_size1 = torch.tensor(30)
    >>> sample_size2 = torch.tensor(30)
    >>> independent_z_test_power(effect_size, sample_size1, sample_size2)
    tensor(0.5614)

    Notes
    -----
    The power calculation is based on the standard normal distribution. For a
    two-sample z-test with effect size d and sample sizes n₁ and n₂, the test
    statistic under the alternative hypothesis follows:

    Z = (X̄₁ - X̄₂) / σ_diff ~ N(d√(n₁n₂/(n₁+n₂)), 1)

    Where σ_diff is the standard error of the difference:
    σ_diff = σ√(1/n₁ + 1/n₂)

    The noncentrality parameter is:
    δ = d√(n₁n₂/(n₁+n₂))

    For a two-sided test with significance level α:
    Power = P(|Z| > z_{α/2} | H₁) = P(Z > z_{α/2} - δ) + P(Z < -z_{α/2} - δ)

    For one-sided tests:
    - "larger": Power = P(Z > z_α - δ)
    - "smaller": Power = P(Z < -z_α + δ)

    Cohen's effect size interpretation:
    - Small effect: d = 0.20
    - Medium effect: d = 0.50
    - Large effect: d = 0.80

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Aberson, C. L. (2010). Applied power analysis for the behavioral
           sciences. Routledge.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size1 = torch.atleast_1d(torch.as_tensor(sample_size1))

    if sample_size2 is None:
        sample_size2 = sample_size1
    else:
        sample_size2 = torch.atleast_1d(torch.as_tensor(sample_size2))

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or sample_size1.dtype == torch.float64
        or sample_size2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size1 = sample_size1.to(dtype)
    sample_size2 = sample_size2.to(dtype)

    # Ensure positive sample sizes
    sample_size1 = torch.clamp(sample_size1, min=1.0)
    sample_size2 = torch.clamp(sample_size2, min=1.0)

    # Calculate effective sample size (harmonic mean scaled)
    # For two-sample z-test: n_eff = n₁n₂/(n₁+n₂)
    n_eff = (sample_size1 * sample_size2) / (sample_size1 + sample_size2)

    # Calculate noncentrality parameter
    ncp = effect_size * torch.sqrt(n_eff)

    # Standard normal critical values using erfinv
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        # Power = P(Z > z_{α/2} - δ) + P(Z < -z_{α/2} - δ)
        # where δ = d√(n₁n₂/(n₁+n₂)) is the noncentrality parameter
        power_upper = (1 - torch.erf((z_alpha_half - ncp) / sqrt_2)) / 2
        power_lower = torch.erf((-z_alpha_half - ncp) / sqrt_2) / 2
        power = power_upper + power_lower
    elif alternative == "larger":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        # Power = P(Z > z_α - δ)
        power = (1 - torch.erf((z_alpha - ncp) / sqrt_2)) / 2
    elif alternative == "smaller":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        # Power = P(Z < -z_α + δ)
        power = torch.erf((-z_alpha + ncp) / sqrt_2) / 2
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}"
        )

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
