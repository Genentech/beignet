import math

import torch
from torch import Tensor


def proportion_two_sample_sample_size(
    p1: Tensor,
    p2: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    *,
    out: Tensor | None = None,
) -> Tensor:
    p1 = torch.atleast_1d(torch.as_tensor(p1))
    p2 = torch.atleast_1d(torch.as_tensor(p2))

    if p1.dtype == torch.float64 or p2.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)

    ratio = torch.tensor(ratio, dtype=dtype)

    epsilon = 1e-8

    p1 = torch.clamp(p1, epsilon, 1 - epsilon)
    p2 = torch.clamp(p2, epsilon, 1 - epsilon)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2

        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    p_pooled = (p1 + p2 * ratio) / (1 + ratio)

    p_pooled = torch.clamp(p_pooled, epsilon, 1 - epsilon)

    effect = torch.abs(p1 - p2)

    effect_safe = torch.where(effect < 1e-6, torch.tensor(1e-6, dtype=dtype), effect)

    var_null = p_pooled * (1 - p_pooled) * (1 + 1 / ratio)

    var_alt = p1 * (1 - p1) + p2 * (1 - p2) / ratio

    numerator = z_alpha * torch.sqrt(var_null) + z_beta * torch.sqrt(var_alt)

    sample_size = (numerator / effect_safe) ** 2

    output = torch.ceil(sample_size)

    output = torch.clamp(output, min=1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
