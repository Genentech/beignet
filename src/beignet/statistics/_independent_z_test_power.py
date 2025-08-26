import math

import torch
from torch import Tensor


def independent_z_test_power(
    effect_size: Tensor,
    sample_size1: Tensor,
    sample_size2: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    sample_size1 = torch.atleast_1d(torch.as_tensor(sample_size1))

    if sample_size2 is None:
        sample_size2 = sample_size1
    else:
        sample_size2 = torch.atleast_1d(torch.as_tensor(sample_size2))

    if (
        effect_size.dtype == torch.float64
        or sample_size1.dtype == torch.float64
        or sample_size2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    sample_size1 = sample_size1.to(dtype)
    sample_size2 = sample_size2.to(dtype)

    sample_size1 = torch.clamp(sample_size1, min=1.0)
    sample_size2 = torch.clamp(sample_size2, min=1.0)

    n_eff = (sample_size1 * sample_size2) / (sample_size1 + sample_size2)

    noncentrality_parameter = effect_size * torch.sqrt(n_eff)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2

        power_upper = (
            1 - torch.erf((z_alpha_half - noncentrality_parameter) / sqrt_2)
        ) / 2
        power_lower = torch.erf((-z_alpha_half - noncentrality_parameter) / sqrt_2) / 2

        power = power_upper + power_lower
    elif alternative == "larger":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        power = (1 - torch.erf((z_alpha - noncentrality_parameter) / sqrt_2)) / 2
    elif alternative == "smaller":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        power = torch.erf((-z_alpha + noncentrality_parameter) / sqrt_2) / 2
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}"
        )

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
