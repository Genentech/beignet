import math

import torch
from torch import Tensor

from ._multivariate_analysis_of_variance_power import (
    multivariate_analysis_of_variance_power,
)


def multivariate_analysis_of_variance_sample_size(
    effect_size: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required total sample size for MANOVA.

    Calculates the total sample size needed to achieve desired power
    for detecting multivariate group differences in MANOVA.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning multi-endpoint clinical trials with adequate power
    - Educational intervention studies with multiple achievement measures
    - Psychological research design with multiple behavioral outcomes
    - Marketing studies comparing products on multiple attributes
    - Quality control studies with multiple measurement criteria

    **Machine Learning Contexts:**
    - Multi-task learning experiments: planning adequate samples for performance comparison across tasks
    - Feature evaluation studies: determining sample sizes for group comparison across multiple features
    - Ensemble method evaluation: planning adequate samples for multi-output model comparison
    - Multi-label classification studies: determining sample sizes for classifier comparison across labels
    - Cross-validation design: planning adequate samples for multi-metric model evaluation
    - A/B testing with multiple KPIs: determining required sample sizes for multivariate effect detection
    - Fairness assessment planning: ensuring adequate power for bias detection across multiple metrics
    - Domain adaptation studies: planning sample sizes for multi-domain, multi-metric comparison
    - Transfer learning evaluation: determining adequate samples for multi-task knowledge transfer assessment
    - Hyperparameter optimization: planning sample sizes for multi-criteria parameter comparison

    **Use this function when:**
    - Planning studies with multiple related dependent variables
    - Expected multivariate effect size can be estimated
    - Multiple groups will be compared simultaneously
    - Sample size is the primary resource constraint
    - Interest is in overall multivariate group differences

    **Sample size considerations:**
    - Required sample size increases with number of variables
    - Required sample size increases with number of groups
    - Larger multivariate effect sizes require smaller sample sizes
    - Balanced group allocation is generally optimal
    - Consider correlation structure among dependent variables

    Parameters
    ----------
    effect_size : Tensor
        Multivariate effect size (multivariate Cohen's f).
    n_variables : Tensor
        Number of dependent variables (p).
    n_groups : Tensor
        Number of groups (k).
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor
        Required total sample size (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> n_variables = torch.tensor(3)
    >>> n_groups = torch.tensor(4)
    >>> multivariate_analysis_of_variance_sample_size(effect_size, n_variables, n_groups)
    tensor(156.0)

    Notes
    -----
    This function uses an iterative approach to find the total sample size
    that achieves the desired power for MANOVA.

    Sample size considerations:
    - More dependent variables require larger sample sizes
    - More groups require larger sample sizes
    - MANOVA is sensitive to violations of multivariate normality
    - Minimum recommended: n > p + k (preferably much larger)
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_variables = torch.atleast_1d(torch.as_tensor(n_variables))
    n_groups = torch.atleast_1d(torch.as_tensor(n_groups))

    dtypes = [effect_size.dtype, n_variables.dtype, n_groups.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_variables = n_variables.to(dtype)
    n_groups = n_groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)
    n_variables = torch.clamp(n_variables, min=1.0)
    n_groups = torch.clamp(n_groups, min=2.0)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    n_init = (
        ((z_alpha + z_beta) / effect_size) ** 2 * n_variables + n_groups + n_variables
    )
    n_init = torch.clamp(n_init, min=n_groups + n_variables + 10)

    n_current = n_init
    for _ in range(15):
        current_power = multivariate_analysis_of_variance_power(
            effect_size, n_current, n_variables, n_groups, alpha=alpha
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.2 * power_gap
        n_current = torch.clamp(
            n_current * adjustment, min=n_groups + n_variables + 10, max=1e6
        )

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
