import math

import torch
from torch import Tensor

from ._jonckheere_terpstra_test_power import jonckheere_terpstra_test_power


def jonckheere_terpstra_test_sample_size(
    effect_size: Tensor,
    groups: Tensor | int,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or groups.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)

    groups = groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)

    groups = torch.clamp(groups, min=3.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    n_initial = ((z_alpha + z_beta) / effect_size) ** 2 / groups

    n_initial = torch.clamp(n_initial, min=5.0)

    n_iteration = n_initial
    for _ in range(12):
        sample_sizes = n_iteration.unsqueeze(-1).expand(
            *n_iteration.shape, int(groups.max().item())
        )

        current_power = jonckheere_terpstra_test_power(
            effect_size, sample_sizes, alpha=alpha
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.3 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=5.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
