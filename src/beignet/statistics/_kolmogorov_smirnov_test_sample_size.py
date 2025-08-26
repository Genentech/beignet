import math

import torch
from torch import Tensor

from ._kolmogorov_smirnov_test_power import kolmogorov_smirnov_test_power


def kolmogorov_smirnov_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    dtype = effect_size.dtype
    if not dtype.is_floating_point:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8, max=1.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alt == "two-sided":
        if alpha == 0.05:
            c_alpha = 1.36
        elif alpha == 0.01:
            c_alpha = 1.63
        else:
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha / 2, dtype=dtype)))
    else:
        if alpha == 0.05:
            c_alpha = 1.22
        elif alpha == 0.01:
            c_alpha = 1.52
        else:
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha, dtype=dtype)))

    sqrt2 = math.sqrt(2.0)

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    n_init = (c_alpha / effect_size) ** 2

    n_init = torch.clamp(n_init, min=10.0)

    adjustment = 1.0 + 0.5 * (z_beta / c_alpha) ** 2

    n_init = n_init * adjustment

    n_current = n_init
    for _ in range(12):
        current_power = kolmogorov_smirnov_test_power(
            effect_size, n_current, alpha=alpha, alternative=alternative
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.5 * power_gap

        n_current = torch.clamp(n_current * adjustment, min=10.0, max=1e6)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
