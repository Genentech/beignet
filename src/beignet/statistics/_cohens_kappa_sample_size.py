import math

import torch
from torch import Tensor

from ._cohens_kappa_power import cohens_kappa_power


def cohens_kappa_sample_size(
    kappa: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size for Cohen's kappa test.

    Calculates the number of items/subjects needed to achieve desired power
    for testing whether Cohen's kappa differs significantly from zero.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning inter-rater reliability studies with adequate power
    - Designing content analysis studies for sufficient coder agreement assessment
    - Clinical trial planning for diagnostic agreement validation
    - Survey research design ensuring reliable categorization measurement
    - Educational assessment planning for grader reliability studies

    **Machine Learning Contexts:**
    - Annotation study design: determining sample sizes for labeler agreement evaluation
    - Active learning planning: ensuring adequate power for annotation quality assessment
    - Model evaluation planning: determining required samples for human-AI agreement testing
    - Fairness audit design: planning sample sizes for demographic agreement studies
    - Cross-validation design: ensuring adequate power for label consistency measurement
    - Ensemble evaluation planning: determining samples needed for model agreement assessment
    - Domain adaptation studies: planning sample sizes for cross-domain label consistency
    - Multi-task learning: determining adequate samples for annotation agreement testing
    - Federated learning design: planning sample sizes for institution annotation consistency
    - Crowdsourcing studies: determining required items for worker reliability assessment

    **Use this function when:**
    - Planning studies where kappa will be the primary agreement measure
    - Expected kappa value can be reasonably estimated from pilot data
    - Two-rater design is planned (not multiple raters)
    - Categorical outcomes with known or estimable category distributions
    - Adequate resources exist to collect the calculated sample size

    **Sample size considerations:**
    - Higher expected kappa values require smaller sample sizes
    - Two-sided tests require larger sample sizes than one-sided
    - Balanced category distributions generally require larger sample sizes
    - Lower alpha levels and higher power requirements increase needed sample size
    - Marginal distributions affect sample size (balanced is often worst case)

    Parameters
    ----------
    kappa : Tensor
        Expected Cohen's kappa coefficient under the alternative hypothesis.
        Range is typically [-1, 1], but often [0, 1] in practice.
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis direction.

    Returns
    -------
    output : Tensor
        Required sample size (number of items/subjects, rounded up).

    Examples
    --------
    >>> kappa = torch.tensor(0.6)  # Substantial agreement
    >>> cohens_kappa_sample_size(kappa)
    tensor(25.0)

    Notes
    -----
    This function uses an iterative approach to find the sample size that
    achieves the desired power for the Cohen's kappa test.

    Sample size considerations:
    - Larger kappa values require smaller sample sizes
    - Two-sided tests require larger sample sizes than one-sided tests
    - The sample size refers to the number of items being rated, not the
      number of raters (which is typically 2)
    """
    kappa = torch.atleast_1d(torch.as_tensor(kappa))

    # Ensure floating point dtype
    dtype = kappa.dtype
    if not dtype.is_floating_point:
        dtype = torch.float32
    kappa = kappa.to(dtype)

    # Validate inputs
    kappa = torch.clamp(kappa, min=-0.99, max=0.99)

    # Initial approximation
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
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Initial guess based on simplified formula
    # Using p_e ≈ 0.5 approximation
    p_e_approx = torch.tensor(0.5, dtype=dtype)
    n_init = ((z_alpha + z_beta) ** 2) * p_e_approx / ((kappa**2) * (1 - p_e_approx))
    n_init = torch.clamp(n_init, min=15.0)

    # Iterative refinement
    n_current = n_init
    for _ in range(12):
        # Calculate current power
        current_power = cohens_kappa_power(
            kappa, n_current, alpha=alpha, alternative=alternative
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.3 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=15.0, max=1e5)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
