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
    icc = torch.atleast_1d(torch.as_tensor(icc))
    n_raters = torch.atleast_1d(torch.as_tensor(n_raters))

    dtype = (
        torch.float64
        if (icc.dtype == torch.float64 or n_raters.dtype == torch.float64)
        else torch.float32
    )
    icc = icc.to(dtype)
    n_raters = n_raters.to(dtype)

    icc = torch.clamp(icc, min=0.01, max=0.99)
    n_raters = torch.clamp(n_raters, min=2.0)

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

    f_expected = (1 + (n_raters - 1) * icc) / (1 - icc)

    effect_size = torch.log(f_expected)
    n_init = ((z_alpha + z_beta) / effect_size) ** 2
    n_init = torch.clamp(n_init, min=10.0)

    n_current = n_init
    for _ in range(15):
        current_power = intraclass_correlation_power(
            icc, n_current, n_raters, alpha=alpha, alternative=alternative
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.4 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=10.0, max=1e5)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
