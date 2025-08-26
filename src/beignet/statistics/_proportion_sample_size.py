import math

import torch
from torch import Tensor


def proportion_sample_size(
    p0: Tensor,
    p1: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    p0 = torch.atleast_1d(torch.as_tensor(p0))
    p1 = torch.atleast_1d(torch.as_tensor(p1))

    if p0.dtype == torch.float64 or p1.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    p0 = p0.to(dtype)
    p1 = p1.to(dtype)

    epsilon = 1e-8
    p0 = torch.clamp(p0, epsilon, 1 - epsilon)
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    se_null = torch.sqrt(p0 * (1 - p0))
    se_alt = torch.sqrt(p1 * (1 - p1))

    effect = torch.abs(p1 - p0)

    effect_safe = torch.where(effect < 1e-6, torch.tensor(1e-6, dtype=dtype), effect)

    sample_size = ((z_alpha * se_null + z_beta * se_alt) / effect_safe) ** 2

    output = torch.clamp(torch.ceil(sample_size), min=1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
