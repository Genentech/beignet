import math

import torch
from torch import Tensor


def correlation_sample_size(
    r: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for correlation tests to achieve specified power.

    Given an expected correlation coefficient, desired power, and significance level,
    this function calculates the minimum sample size needed.

    This function is differentiable with respect to the r parameter.
    While traditional sample size calculations don't require gradients,
    differentiability can be useful when correlation strengths are learned
    parameters or when optimizing experimental designs in machine learning contexts.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning studies to detect associations between continuous variables
    - Validating measurement instruments for convergent/discriminant validity
    - Survey research planning to detect relationships between scale scores
    - Quality control: planning studies to detect process variable correlations
    - Medical research: planning studies to detect biomarker associations
    - Educational research: planning studies to detect test score relationships

    **Machine Learning Contexts:**
    - Feature selection: planning studies to detect feature-target correlations
    - Model evaluation: planning validation of feature importance relationships
    - Hyperparameter optimization: planning correlation studies between parameters and performance
    - Cross-validation: planning studies to detect stability correlations across folds
    - Transfer learning: planning studies to detect source-target domain correlations
    - Ensemble methods: planning studies to detect base model correlation patterns
    - Active learning: planning studies to detect query strategy effectiveness correlations
    - Domain adaptation: planning studies to detect adaptation metric correlations
    - Fairness evaluation: planning studies to detect bias correlations in model outputs
    - Causal inference: planning studies to detect instrument variable correlations
    - Recommendation systems: planning studies to detect user-item correlation patterns
    - Time series: planning studies to detect temporal correlation patterns

    **Choose correlation tests over other tests when:**
    - Both variables are continuous (or can be meaningfully treated as such)
    - Interest is in linear association strength rather than group differences
    - Need to quantify degree of relationship rather than just test for association
    - Working within correlation/regression framework

    **Choose Pearson correlation over other correlation measures when:**
    - Both variables are approximately normally distributed
    - Relationship is expected to be linear
    - Working with interval or ratio scale measurements
    - Need parametric confidence intervals and significance tests

    **Interpretation Guidelines:**
    - Effect size r: 0.10 (small), 0.30 (medium), 0.50 (large) correlations
    - Sample size increases dramatically as expected correlation approaches zero
    - Fisher z-transformation ensures normal sampling distribution
    - Two-sided tests require larger samples than one-sided tests
    - Consider nonlinear relationships that might affect linear correlation
    - Account for restriction of range that can attenuate correlations

    Parameters
    ----------
    r : Tensor
        Expected correlation coefficient. Can be a scalar or tensor.

    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> r = torch.tensor(0.3)
    >>> correlation_sample_size(r, power=0.8)
    tensor(85)
    """
    # Convert inputs to tensors if needed
    r = torch.atleast_1d(torch.as_tensor(r))

    # Fisher z-transformation of the correlation
    # z_r = 0.5 * ln((1 + r) / (1 - r))
    epsilon = 1e-7  # Small value to avoid division by zero
    r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)
    z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=r.dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=r.dtype)) * sqrt_2
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=r.dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=r.dtype)) * sqrt_2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Sample size formula for correlation test
    # For Fisher z-transform: SE = 1/sqrt(n-3), so n = ((z_alpha + z_beta) / z_r)^2 + 3
    # This comes from the power calculation: z_r / (1/sqrt(n-3)) = z_alpha + z_beta

    # Avoid division by very small correlations
    z_r_safe = torch.where(torch.abs(z_r) < 1e-6, torch.sign(z_r) * 1e-6, z_r)

    sample_size = ((z_alpha + z_beta) / torch.abs(z_r_safe)) ** 2 + 3

    # Round up to nearest integer
    output = torch.ceil(sample_size)

    if out is not None:
        out.copy_(output)
        return out

    return output
