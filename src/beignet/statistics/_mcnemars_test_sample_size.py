import math

import torch
from torch import Tensor

from ._mcnemars_test_power import mcnemars_test_power


def mcnemars_test_sample_size(
    p01: Tensor,
    p10: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))

    dtype = (
        torch.float64
        if (p01.dtype == torch.float64 or p10.dtype == torch.float64)
        else torch.float32
    )
    p01 = p01.to(dtype)
    p10 = p10.to(dtype)

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        pt = torch.tensor(prob, dtype=dtype)

        eps = torch.finfo(dtype).eps

        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if two_sided:
        z_alpha = z_of(1 - alpha / 2)
    else:
        z_alpha = z_of(1 - alpha)

    z_beta = z_of(power)

    probability = torch.where(
        (p01 + p10) > 0,
        p01 / torch.clamp(p01 + p10, min=1e-12),
        torch.zeros_like(p01),
    )
    delta = torch.abs(probability - 0.5)

    n0 = ((z_alpha + z_beta) / (2 * torch.clamp(delta, min=1e-8))) ** 2 / torch.clamp(
        p01 + p10,
        min=1e-8,
    )
    n0 = torch.clamp(n0, min=4.0)

    n_curr = n0
    for _ in range(12):
        pwr = mcnemars_test_power(
            p01,
            p10,
            torch.ceil(n_curr),
            alpha=alpha,
            two_sided=two_sided,
        )
        gap = torch.clamp(power - pwr, min=-0.45, max=0.45)

        n_curr = torch.clamp(n_curr * (1.0 + 1.25 * gap), min=4.0, max=1e7)

    n_out = torch.ceil(n_curr)
    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
    return result
