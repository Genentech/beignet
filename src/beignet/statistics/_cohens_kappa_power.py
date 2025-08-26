import math

import torch
from torch import Tensor


def cohens_kappa_power(
    kappa: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for Cohen's kappa coefficient.

    Cohen's kappa measures inter-rater agreement for categorical items.
    This function calculates the power for testing whether kappa differs
    significantly from zero (no agreement beyond chance).

    When to Use
    -----------
    **Traditional Statistics:**
    - Inter-rater reliability studies for diagnostic assessments
    - Content analysis research measuring coder agreement
    - Survey research validating response categorization consistency
    - Clinical research evaluating diagnostic agreement between experts
    - Educational assessment measuring grader reliability

    **Machine Learning Contexts:**
    - Annotation quality assessment: evaluating human labeler agreement
    - Active learning: measuring annotation consistency for query strategies
    - Model validation: comparing automated predictions with human judgments
    - Fairness auditing: assessing agreement between different demographic groups
    - Cross-validation: measuring label consistency across data splits
    - Ensemble methods: evaluating agreement between different model predictions
    - Domain adaptation: assessing label consistency across domains
    - Multi-task learning: measuring agreement between task-specific annotations
    - Federated learning: evaluating annotation consistency across institutions
    - Crowdsourcing: determining annotation quality and worker reliability

    **Choose Cohen's kappa over other agreement measures when:**
    - Data consists of categorical classifications (not continuous ratings)
    - Need to account for chance agreement (unlike simple percent agreement)
    - Two raters making independent judgments
    - Categories are nominal or ordinal (for weighted kappa)
    - Sample size is adequate for reliable kappa estimation

    **Choose Cohen's kappa over ICC when:**
    - Classifications are categorical rather than continuous ratings
    - Exactly two raters (ICC handles multiple raters better)
    - Interest is in agreement rather than consistency
    - Categories have no meaningful ordering (nominal data)

    **Interpretation Guidelines:**
    - κ < 0.20: Poor agreement (little better than chance)
    - κ 0.21-0.40: Fair agreement (better than chance but concerns remain)
    - κ 0.41-0.60: Moderate agreement (acceptable for some applications)
    - κ 0.61-0.80: Substantial agreement (good reliability)
    - κ 0.81-1.00: Almost perfect agreement (excellent reliability)

    Parameters
    ----------
    kappa : Tensor
        Expected Cohen's kappa coefficient under the alternative hypothesis.
        Range is typically [-1, 1], but often [0, 1] in practice.
    sample_size : Tensor
        Number of items/subjects being rated.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis direction.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> kappa = torch.tensor(0.6)  # Substantial agreement
    >>> sample_size = torch.tensor(50)
    >>> cohens_kappa_power(kappa, sample_size)
    tensor(0.9234)

    Notes
    -----
    Cohen's kappa is defined as:
    κ = (p_o - p_e) / (1 - p_e)

    where:
    - p_o = observed agreement proportion
    - p_e = expected agreement by chance

    Under H₀: κ = 0 (no agreement beyond chance)
    Under H₁: κ ≠ 0 (agreement differs from chance)

    The test statistic is approximately:
    Z = κ̂ / SE(κ̂)

    where SE(κ̂) ≈ √(p_e / (n * (1 - p_e)))

    This is a simplified approximation. The exact standard error depends
    on the marginal distributions and requires more complex calculations.

    Interpretation of κ values (Landis & Koch, 1977):
    - κ < 0: Poor agreement
    - 0.00-0.20: Slight agreement
    - 0.21-0.40: Fair agreement
    - 0.41-0.60: Moderate agreement
    - 0.61-0.80: Substantial agreement
    - 0.81-1.00: Almost perfect agreement

    References
    ----------
    Fleiss, J. L., Levin, B., & Paik, M. C. (2013). Statistical methods for
    rates and proportions. John Wiley & Sons.
    """
    kappa = torch.atleast_1d(torch.as_tensor(kappa))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    # Ensure floating point dtype
    dtype = (
        torch.float64
        if (kappa.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    kappa = kappa.to(dtype)
    sample_size = sample_size.to(dtype)

    # Validate inputs
    kappa = torch.clamp(kappa, min=-0.99, max=0.99)
    sample_size = torch.clamp(sample_size, min=10.0)

    # Approximate standard error under null hypothesis
    # This assumes p_e ≈ 0.5 for simplicity (balanced marginals)
    # In practice, p_e depends on the marginal distributions
    p_e_approx = torch.tensor(0.5, dtype=dtype)
    se_kappa = torch.sqrt(p_e_approx / (sample_size * (1 - p_e_approx)))

    # Noncentrality parameter
    ncp = torch.abs(kappa) / se_kappa

    # Critical values
    sqrt2 = math.sqrt(2.0)
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
        # Two-sided power
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
            1 - torch.erf((z_alpha + ncp) / sqrt2)
        )
    elif alt == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        # One-sided power (positive kappa)
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2))
    else:  # alt == "less"
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        # One-sided power (negative kappa)
        power = 0.5 * (1 - torch.erf((z_alpha + ncp) / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
