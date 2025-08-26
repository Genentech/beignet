import math

import torch
from torch import Tensor


def kolmogorov_smirnov_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0, max=1.0)

    sample_size = torch.clamp(sample_size, min=3.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt_n = torch.sqrt(sample_size)

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

    d_critical = c_alpha / sqrt_n

    expected_d = effect_size

    se_d = torch.sqrt(1.0 / (2 * sample_size))

    z_score = (d_critical - expected_d) / torch.clamp(se_d, min=1e-12)

    square_root_two = math.sqrt(2.0)
    if alt == "two-sided":
        power = 1 - torch.erf(torch.abs(z_score) / square_root_two)
    elif alt == "greater":
        power = 0.5 * (1 - torch.erf(z_score / square_root_two))
    else:
        power = 0.5 * (1 + torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
