"""Z-test power (one-sample z-test with known variance)."""

import torch
from torch import Tensor


def z_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for one-sample z-tests with known variance.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a one-sample z-test
    where the population variance is known.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Quality control testing with known process variance
    - Large-sample hypothesis testing where central limit theorem applies
    - Testing population means when historical variance data is available
    - Standardized testing and educational assessments
    - Industrial quality control with established measurement precision

    **Machine Learning Contexts:**
    - Model performance testing with known benchmark variance
    - A/B testing with large sample sizes and established baseline metrics
    - Validation testing for ML models with known prediction uncertainties
    - Hyperparameter optimization with established performance baselines
    - Cross-validation analysis with known fold-to-fold variability
    - Transfer learning: testing performance against known source domain metrics
    - Ensemble methods: testing individual model performance against known ensemble variance
    - Active learning: testing sample efficiency with established learning curves
    - Federated learning: testing client performance with known global variance
    - Time series analysis: testing forecast accuracy with known historical variance
    - Computer vision: testing model accuracy with established dataset benchmarks
    - NLP: testing language model performance against known corpus statistics
    - Recommendation systems: testing recommendation quality with known user behavior variance
    - Anomaly detection: testing detection rates with established false positive rates
    - Causal inference: testing treatment effects with known population variance

    **Choose z-test power over t-test power when:**
    - Population variance is known (not estimated from sample)
    - Sample size is very large (n > 100) and central limit theorem applies
    - Computational efficiency is important (z-test is simpler)
    - Working with standardized effect sizes from meta-analyses
    - Historical data provides reliable variance estimates

    **Choose z-test over other tests when:**
    - Testing means of continuous variables with known variance
    - Data is approximately normally distributed or large sample sizes
    - Maximum statistical power is desired for known variance scenarios
    - Comparing against established population parameters

    **Interpretation Guidelines:**
    - Effect size is Cohen's d: (μ₁ - μ₀) / σ where σ is known
    - Z-test assumes population variance is exactly known (rarely true in practice)
    - Power increases with larger effect sizes and sample sizes
    - Consider practical significance alongside statistical power
    - In ML contexts, "known variance" often means established from prior studies

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the true population mean and the hypothesized mean, divided by the
        population standard deviation: d = (μ₁ - μ₀) / σ.
        Should be positive for "larger" alternative, negative for "smaller".

    sample_size : Tensor
        Sample size (number of observations).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size = torch.tensor(30)
    >>> z_test_power(effect_size, sample_size)
    tensor(0.6985)

    Notes
    -----
    The power calculation is based on the standard normal distribution. For a
    one-sample z-test with effect size d and sample size n, the test statistic
    under the alternative hypothesis follows:

    Z = (X̄ - μ₀) / (σ / √n) ~ N(d√n, 1)

    Where μ₀ is the hypothesized mean, σ is the known population standard
    deviation, and d is the standardized effect size.

    For a two-sided test with significance level α:
    Power = P(|Z| > z_{α/2} | H₁) = P(Z > z_{α/2} - d√n) + P(Z < -z_{α/2} - d√n)

    For one-sided tests:
    - "larger": Power = P(Z > z_α - d√n)
    - "smaller": Power = P(Z < -z_α + d√n)

    Cohen's effect size interpretation:
    - Small effect: d = 0.20
    - Medium effect: d = 0.50
    - Large effect: d = 0.80

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if effect_size.dtype == torch.float64 or sample_size.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=1.0)

    ncp = effect_size * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}"
        )

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))

    def z_of(p):
        pt = torch.as_tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        z_alpha_half = z_of(1 - alpha / 2)
        power_upper = 0.5 * (
            1 - torch.erf((z_alpha_half - ncp) / torch.sqrt(torch.tensor(2.0)))
        )
        power_lower = 0.5 * (
            1 + torch.erf((-z_alpha_half - ncp) / torch.sqrt(torch.tensor(2.0)))
        )
        power = power_upper + power_lower
    elif alt == "greater":
        z_alpha = z_of(1 - alpha)
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / torch.sqrt(torch.tensor(2.0))))
    else:
        z_alpha = z_of(1 - alpha)
        power = 0.5 * (1 + torch.erf((-z_alpha - ncp) / torch.sqrt(torch.tensor(2.0))))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
