import math

import torch
from torch import Tensor

from ._friedman_test_power import friedman_test_power


def friedman_test_sample_size(
    effect_size: Tensor,
    n_treatments: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    n_treatments = torch.atleast_1d(torch.as_tensor(n_treatments))

    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or n_treatments.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)

    n_treatments = n_treatments.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)

    n_treatments = torch.clamp(n_treatments, min=3.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    n_init = (
        ((z_alpha + z_beta) ** 2)
        * n_treatments
        * (n_treatments + 1)
        / (12 * effect_size)
    )
    n_init = torch.clamp(n_init, min=5.0)

    n_current = n_init
    for _ in range(12):
        current_power = friedman_test_power(
            effect_size, n_current, n_treatments, alpha=alpha
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_current = torch.clamp(n_current * adjustment, min=5.0, max=1e6)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
