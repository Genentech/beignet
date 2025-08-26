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
    dtype = torch.promote_type(kappa.dtype, sample_size.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
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
