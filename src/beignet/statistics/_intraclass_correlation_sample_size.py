import math

import torch
from torch import Tensor

from ._intraclass_correlation_power import intraclass_correlation_power


def intraclass_correlation_sample_size(
    icc: Tensor,
    n_raters: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "greater",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required number of subjects for ICC study.

    Calculates the number of subjects needed to achieve desired power
    for testing whether the ICC differs significantly from zero.

    Parameters
    ----------
    icc : Tensor
        Expected ICC under the alternative hypothesis.
        Range is [0, 1] for most ICC models.
    n_raters : Tensor
        Number of raters or repeated measurements per subject.
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="greater"
        Alternative hypothesis direction. "greater" is most common for ICC.

    Returns
    -------
    output : Tensor
        Required number of subjects (rounded up to nearest integer).

    Examples
    --------
    >>> icc = torch.tensor(0.7)  # Good reliability
    >>> n_raters = torch.tensor(3)
    >>> intraclass_correlation_sample_size(icc, n_raters)
    tensor(18.0)

    Notes
    -----
    This function uses an iterative approach to find the number of subjects
    that achieves the desired power for the ICC test.

    Sample size considerations:
    - Higher ICC values require fewer subjects
    - More raters per subject reduce the required number of subjects
    - The number of subjects is often the limiting factor in ICC studies
    """
    icc = torch.atleast_1d(torch.as_tensor(icc))
    n_raters = torch.atleast_1d(torch.as_tensor(n_raters))

    # Ensure floating point dtype
    dtype = torch.promote_type(icc.dtype, n_raters.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    icc = icc.to(dtype)
    n_raters = n_raters.to(dtype)

    # Validate inputs
    icc = torch.clamp(icc, min=0.01, max=0.99)
    n_raters = torch.clamp(n_raters, min=2.0)

    # Initial approximation based on F-test power formula
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

    # Expected F-ratio under alternative
    f_expected = (1 + (n_raters - 1) * icc) / (1 - icc)

    # Initial guess based on effect size
    effect_size = torch.log(f_expected)
    n_init = ((z_alpha + z_beta) / effect_size) ** 2
    n_init = torch.clamp(n_init, min=10.0)

    # Iterative refinement
    n_current = n_init
    for _ in range(15):
        # Calculate current power
        current_power = intraclass_correlation_power(
            icc, n_current, n_raters, alpha=alpha, alternative=alternative
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.4 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=10.0, max=1e5)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
