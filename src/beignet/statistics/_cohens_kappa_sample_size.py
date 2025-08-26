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

    p_e_approximate = torch.tensor(0.5, dtype=dtype)

    n_initial = (
        ((z_alpha + z_beta) ** 2)
        * p_e_approximate
        / ((kappa**2) * (1 - p_e_approximate))
    )

    n_initial = torch.clamp(n_initial, min=15.0)

    n_iteration = n_initial
    for _ in range(12):
        current_power = cohens_kappa_power(
            kappa, n_iteration, alpha=alpha, alternative=alternative
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.3 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=15.0, max=1e5)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
