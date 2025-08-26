import math

import torch
from torch import Tensor


def correlation_sample_size(
    r: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r = torch.atleast_1d(torch.as_tensor(r))

    epsilon = 1e-7

    r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)

    z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=r.dtype)) * sqrt_2

        z_beta = torch.erfinv(torch.tensor(power, dtype=r.dtype)) * sqrt_2
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=r.dtype)) * sqrt_2

        z_beta = torch.erfinv(torch.tensor(power, dtype=r.dtype)) * sqrt_2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    z_r_safe = torch.where(torch.abs(z_r) < 1e-6, torch.sign(z_r) * 1e-6, z_r)

    sample_size = ((z_alpha + z_beta) / torch.abs(z_r_safe)) ** 2 + 3

    output = torch.ceil(sample_size)

    if out is not None:
        out.copy_(output)
        return out

    return output
