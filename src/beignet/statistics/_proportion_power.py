import math

import torch
from torch import Tensor


def proportion_power(
    p0: Tensor,
    p1: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    p0 = torch.atleast_1d(torch.as_tensor(p0))
    p1 = torch.atleast_1d(torch.as_tensor(p1))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if (
        p0.dtype == torch.float64
        or p1.dtype == torch.float64
        or sample_size.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    p0 = p0.to(dtype)
    p1 = p1.to(dtype)
    sample_size = sample_size.to(dtype)

    epsilon = 1e-8
    p0 = torch.clamp(p0, epsilon, 1 - epsilon)
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)

    sqrt_2 = math.sqrt(2.0)

    se_null = torch.sqrt(p0 * (1 - p0) / sample_size)

    se_alt = torch.sqrt(p1 * (1 - p1) / sample_size)

    effect = (p1 - p0) / se_null

    if alternative == "two-sided":
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2

        variance_ratio = se_alt / se_null
        adjusted_effect = effect / variance_ratio

        power = (1 - torch.erf((z_alpha_half - adjusted_effect) / sqrt_2)) / 2 + (
            1 - torch.erf((z_alpha_half + adjusted_effect) / sqrt_2)
        ) / 2

    elif alternative == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        variance_ratio = se_alt / se_null
        adjusted_effect = effect / variance_ratio

        power = (1 - torch.erf((z_alpha - adjusted_effect) / sqrt_2)) / 2

    elif alternative == "less":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        variance_ratio = se_alt / se_null
        adjusted_effect = effect / variance_ratio

        power = (1 + torch.erf((-z_alpha - adjusted_effect) / sqrt_2)) / 2

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
