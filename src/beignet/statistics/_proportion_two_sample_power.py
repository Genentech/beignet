import math

import torch
from torch import Tensor


def proportion_two_sample_power(
    p1: Tensor,
    p2: Tensor,
    n1: Tensor,
    n2: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    p1 = torch.atleast_1d(torch.as_tensor(p1))
    p2 = torch.atleast_1d(torch.as_tensor(p2))

    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(n1))

    if n2 is None:
        sample_size_group_2 = sample_size_group_1
    else:
        sample_size_group_2 = torch.atleast_1d(torch.as_tensor(n2))

    if (
        p1.dtype == torch.float64
        or p2.dtype == torch.float64
        or sample_size_group_1.dtype == torch.float64
        or sample_size_group_2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)

    sample_size_group_1 = sample_size_group_1.to(dtype)
    sample_size_group_2 = sample_size_group_2.to(dtype)

    epsilon = 1e-8

    p1 = torch.clamp(p1, epsilon, 1 - epsilon)
    p2 = torch.clamp(p2, epsilon, 1 - epsilon)

    p_pooled = (sample_size_group_1 * p1 + sample_size_group_2 * p2) / (
        sample_size_group_1 + sample_size_group_2
    )
    p_pooled = torch.clamp(p_pooled, epsilon, 1 - epsilon)

    se_null = torch.sqrt(
        p_pooled * (1 - p_pooled) * (1 / sample_size_group_1 + 1 / sample_size_group_2)
    )

    se_alt = torch.sqrt(
        p1 * (1 - p1) / sample_size_group_1 + p2 * (1 - p2) / sample_size_group_2
    )

    effect = (p1 - p2) / se_null

    sqrt_2 = math.sqrt(2.0)

    effect = p1 - p2

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2

        standardized_effect = torch.abs(effect) / se_alt

        power = (1 - torch.erf((z_alpha - standardized_effect) / sqrt_2)) / 2 + (
            1 - torch.erf((z_alpha + standardized_effect) / sqrt_2)
        ) / 2

    elif alternative == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        standardized_effect = effect / se_alt

        power = (1 - torch.erf((z_alpha - standardized_effect) / sqrt_2)) / 2

    elif alternative == "less":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        standardized_effect = effect / se_alt

        power = (1 + torch.erf((-z_alpha - standardized_effect) / sqrt_2)) / 2

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
