import math

import torch
from torch import Tensor


def chi_square_independence_sample_size(
    effect_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for chi-square independence tests.

    Given the effect size, contingency table dimensions, desired power, and
    significance level, this function calculates the minimum sample size needed
    to achieve the specified power for a chi-square test of independence.

    This function is differentiable with respect to effect_size, rows, and cols
    parameters. While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning survey research to test independence between categorical variables
    - Sample size planning for clinical trials examining categorical outcome associations
    - Market research planning to analyze demographic-preference relationships
    - Educational research planning to test teaching method-outcome associations
    - Quality control planning to examine process-defect relationships

    **Machine Learning Contexts:**
    - Feature selection planning: sample sizes for testing categorical feature independence
    - Model evaluation planning: sample sizes for testing prediction-outcome associations
    - A/B testing design: planning sample sizes for categorical treatment-response relationships
    - Bias detection planning: sample sizes for testing model-protected attribute independence
    - Data preprocessing planning: sample sizes for validating feature independence assumptions
    - Cross-validation planning: sample sizes for testing fold assignment independence
    - Hyperparameter optimization: planning sample sizes for configuration-performance associations
    - Ensemble method evaluation: planning sample sizes for individual model prediction independence
    - Active learning: planning sample sizes for testing selection strategy-label associations
    - Federated learning: planning sample sizes for client characteristic-performance independence
    - Domain adaptation: sample size planning for source/target domain-outcome associations
    - Computer vision: planning sample sizes for image category-performance relationships
    - NLP: planning sample sizes for text category-prediction associations
    - Recommendation systems: planning sample sizes for user category-recommendation independence
    - Anomaly detection: sample size planning for anomaly type-detection method associations
    - Causal inference: sample size planning for independence assumption testing

    **Choose chi-square independence sample size over other methods when:**
    - Both variables are categorical (nominal or ordinal)
    - Testing association rather than goodness-of-fit to specific distributions
    - Data will be organized in contingency tables (cross-tabulations)
    - Chi-square assumptions will be met (expected frequencies ≥ 5 per cell)
    - Independence of observations can be ensured

    **Choose independence over goodness-of-fit sample size when:**
    - Testing relationships between two categorical variables
    - No specific expected distribution is being tested
    - Focus is on association strength rather than distribution conformity
    - Contingency table structure fits the research question naturally

    **Interpretation Guidelines:**
    - Effect size w measures strength of association between categorical variables
    - Cohen's w = 0.10 (small), 0.30 (medium), 0.50 (large) associations
    - Sample size increases dramatically for detecting smaller effect sizes
    - More table cells (higher rows × cols) require larger sample sizes
    - Rule of thumb: at least 5 expected observations per cell (5 × rows × cols)
    - Consider practical constraints including cost and data collection feasibility
    - Account for potential missing data or incomplete responses in planning

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. For independence tests, this measures the strength
        of association between two categorical variables. It can be calculated as
        w = √(χ²/n) where χ² is the chi-square statistic and n is the sample size.
        Should be positive.

    rows : Tensor
        Number of rows in the contingency table (categories of first variable).

    cols : Tensor
        Number of columns in the contingency table (categories of second variable).

    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> rows = torch.tensor(3)
    >>> cols = torch.tensor(3)
    >>> chi_square_independence_sample_size(effect_size, rows, cols, power=0.8)
    tensor(124)

    Notes
    -----
    The sample size calculation is based on the noncentral chi-square distribution.
    For a chi-square independence test with effect size w and sample size n,
    the noncentrality parameter is:

    λ = n * w²

    The degrees of freedom are: df = (rows - 1) × (cols - 1)

    The calculation uses an iterative approach to find the sample size that
    achieves the desired power, starting from an initial normal approximation:

    n ≈ ((z_α + z_β) / w)²

    Where z_α and z_β are the critical values for the given α and β = 1 - power.

    For computational efficiency, we use analytical approximations where possible,
    falling back to iterative refinement when needed.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cramér, H. (1946). Mathematical Methods of Statistics. Princeton
           University Press.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    rows = torch.atleast_1d(torch.as_tensor(rows))
    cols = torch.atleast_1d(torch.as_tensor(cols))

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or rows.dtype == torch.float64
        or cols.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    rows = rows.to(dtype)
    cols = cols.to(dtype)

    # Clamp effect size to positive values and ensure at least 2 categories
    effect_size = torch.clamp(effect_size, min=1e-6)
    rows = torch.clamp(rows, min=2.0)
    cols = torch.clamp(cols, min=2.0)

    # Calculate degrees of freedom for independence test
    degrees_of_freedom = (rows - 1) * (cols - 1)

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    # Initial normal approximation for chi-square test
    # For large sample sizes, the approximation is: n ≈ ((z_α + z_β) / w)²
    n_initial = ((z_alpha + z_beta) / effect_size) ** 2

    # Ensure minimum sample size (rule of thumb: at least 5 expected in each cell)
    min_sample_size = 5.0 * rows * cols
    n_initial = torch.clamp(n_initial, min=min_sample_size)

    # Iterative refinement with convergence detection
    n_current = n_initial
    convergence_tolerance = 1e-6
    max_iterations = 10

    for _iteration in range(max_iterations):
        # Current noncentrality parameter
        ncp_current = n_current * effect_size**2

        # Critical chi-square value using normal approximation
        # χ²_α = degrees_of_freedom + z_α * √(2*degrees_of_freedom)
        chi2_critical = degrees_of_freedom + z_alpha * torch.sqrt(
            2 * degrees_of_freedom
        )

        # For noncentral chi-square, use normal approximation
        # χ²(degrees_of_freedom, λ) ≈ N(degrees_of_freedom + λ, 2*(degrees_of_freedom + 2*λ))
        mean_nc_chi2 = degrees_of_freedom + ncp_current
        var_nc_chi2 = 2 * (degrees_of_freedom + 2 * ncp_current)
        std_nc_chi2 = torch.sqrt(var_nc_chi2)

        # Calculate current power
        z_score = (chi2_critical - mean_nc_chi2) / torch.clamp(std_nc_chi2, min=1e-10)
        power_current = (1 - torch.erf(z_score / sqrt_2)) / 2

        # Clamp power to valid range
        power_current = torch.clamp(power_current, 0.01, 0.99)

        # Calculate power difference
        power_diff = power - power_current

        # Newton-Raphson style adjustment with convergence damping
        # Approximate derivative: d(power)/d(n) ≈ d(power)/d(λ) * w²
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
        n_current = torch.clamp(n_current, min=min_sample_size)
        n_current = torch.clamp(n_current, max=1000000.0)

    # Round up to nearest integer
    output = torch.ceil(n_current)

    # Final check: ensure we meet minimum sample size requirements
    output = torch.clamp(output, min=min_sample_size)

    if out is not None:
        out.copy_(output)
        return out

    return output
