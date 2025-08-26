import math

import torch
from torch import Tensor


def paired_t_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    dtype = torch.float64 if effect_size.dtype == torch.float64 else torch.float32
    effect_size = torch.clamp(effect_size.to(dtype), min=1e-8)

    sqrt2 = math.sqrt(2.0)
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    sample_size = ((z_alpha + z_beta) / effect_size) ** 2
    sample_size = torch.clamp(sample_size, min=2.0)

    sample_size_curr = sample_size
    for _ in range(10):
        degrees_of_freedom = torch.clamp(sample_size_curr - 1, min=1.0)
        tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
        noncentrality_parameter = effect_size * torch.sqrt(sample_size_curr)
        var_nct = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + noncentrality_parameter**2)
            / (degrees_of_freedom - 2),
            1
            + noncentrality_parameter**2
            / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
        )
        std_nct = torch.sqrt(var_nct)
        if alt == "two-sided":
            zu = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            zl = (-tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            current_power = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            current_power = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-tcrit - noncentrality_parameter) / torch.clamp(
                std_nct, min=1e-10
            )
            current_power = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        gap = torch.clamp(power - current_power, min=-0.45, max=0.45)
        sample_size_curr = torch.clamp(
            sample_size_curr * (1.0 + 1.25 * gap), min=2.0, max=1e7
        )

    sample_size_out = torch.ceil(sample_size_curr)
    if out is not None:
        out.copy_(sample_size_out)
        return out
    return sample_size_out
