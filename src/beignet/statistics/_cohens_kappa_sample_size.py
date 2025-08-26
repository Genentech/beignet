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
    kappa = torch.atleast_1d(torch.as_tensor(kappa))

    dtype = kappa.dtype
    if not dtype.is_floating_point:
        dtype = torch.float32
    kappa = kappa.to(dtype)

    kappa = torch.clamp(kappa, min=-0.99, max=0.99)

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

    p_e_approx = torch.tensor(0.5, dtype=dtype)
    n_init = ((z_alpha + z_beta) ** 2) * p_e_approx / ((kappa**2) * (1 - p_e_approx))
    n_init = torch.clamp(n_init, min=15.0)

    n_current = n_init
    for _ in range(12):
        current_power = cohens_kappa_power(
            kappa, n_current, alpha=alpha, alternative=alternative
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.3 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=15.0, max=1e5)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
