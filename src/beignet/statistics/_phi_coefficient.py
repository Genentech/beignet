import torch
from torch import Tensor


def phi_coefficient(
    chi_square: Tensor,
    sample_size: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute phi coefficient effect size for 2×2 chi-square tests.

    The phi coefficient is a measure of association for two binary variables,
    equivalent to the Pearson correlation coefficient when both variables are
    binary. It ranges from -1 to +1, where 0 indicates no association.

    This function is differentiable with respect to both input parameters.
    While differentiability isn't typically needed for effect size calculations,
    it enables integration into machine learning pipelines where chi-square
    statistics may be computed from learned representations.

    When to Use
    -----------
    **Traditional Statistics:**
    - Measuring association between two binary variables (2×2 contingency tables)
    - Medical research: association between treatment and outcome (success/failure)
    - Educational assessment: relationship between binary test results
    - Quality control: association between process factors and pass/fail outcomes
    - Survey research: relationship between binary demographic and response variables

    **Machine Learning Contexts:**
    - Feature selection: measuring association between binary features and binary targets
    - Model evaluation: relationship between binary predictions and true binary labels
    - A/B testing: measuring association strength in binary outcome experiments
    - Fairness auditing: measuring association between protected attributes and decisions
    - Classification performance: alternative to other binary classification metrics
    - Data preprocessing: identifying strongly associated binary feature pairs
    - Anomaly detection: measuring association between binary flags and anomaly status
    - Recommendation systems: measuring user preference associations (like/dislike)
    - Computer vision: measuring association between binary image features
    - NLP: measuring association between binary text features and sentiment
    - Web analytics: measuring association between user actions and conversions
    - Causal inference: measuring association in binary treatment-outcome studies

    **Choose phi coefficient over other measures when:**
    - Both variables are genuinely binary (not artificially dichotomized)
    - Working specifically with 2×2 contingency tables
    - Need a correlation-like measure for binary data (-1 to +1 range)
    - Comparing with Pearson correlations on the same binary variables
    - Effect size interpretation needs to follow correlation conventions

    **Choose phi over Cramer's V when:**
    - Data is exactly 2×2 (Cramer's V reduces to |phi| for 2×2 tables)
    - Need directional information (phi can be negative, Cramer's V is always positive)
    - Working within correlation framework where sign matters

    **Interpretation Guidelines:**
    - |φ| = 0.10 (small effect), 0.30 (medium effect), 0.50 (large effect)
    - Ranges from -1 to +1, similar to Pearson correlation
    - Sign depends on table arrangement (which category is coded as 1)
    - Values near ±1 indicate very strong association between binary variables
    - φ² gives proportion of variance shared between the two binary variables

    Parameters
    ----------
    chi_square : Tensor
        Chi-square test statistic from a 2×2 contingency table.

    sample_size : Tensor
        Total sample size (sum of all cells in the contingency table).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Phi coefficient (between -1 and 1).

    Examples
    --------
    >>> chi_sq = torch.tensor(6.25)
    >>> n = torch.tensor(100)
    >>> phi_coefficient(chi_sq, n)
    tensor(0.2500)

    Notes
    -----
    The phi coefficient is only appropriate for 2×2 contingency tables.
    For larger tables, use Cramer's V instead.

    The interpretation guidelines (Cohen, 1988):
    - Small effect: |φ| = 0.10
    - Medium effect: |φ| = 0.30
    - Large effect: |φ| = 0.50

    The sign of phi depends on the arrangement of the data in the contingency
    table and indicates the direction of association.
    """
    chi_square = torch.atleast_1d(torch.as_tensor(chi_square))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if chi_square.dtype != sample_size.dtype:
        if chi_square.dtype == torch.float64 or sample_size.dtype == torch.float64:
            chi_square = chi_square.to(torch.float64)
            sample_size = sample_size.to(torch.float64)
        else:
            chi_square = chi_square.to(torch.float32)
            sample_size = sample_size.to(torch.float32)

    output = torch.sqrt(chi_square / sample_size)

    output = torch.clamp(output, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
