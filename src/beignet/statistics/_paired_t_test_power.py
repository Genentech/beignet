import math

import torch
from torch import Tensor


def paired_t_test_power(
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
    sample_size = torch.clamp(sample_size.to(dtype), min=2.0)

    degrees_of_freedom = sample_size - 1
    noncentrality_parameter = effect_size * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
    else:
        z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    var_nct = torch.where(
        degrees_of_freedom > 2,
        (degrees_of_freedom + noncentrality_parameter**2) / (degrees_of_freedom - 2),
        1 + noncentrality_parameter**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
    )
    std_nct = torch.sqrt(var_nct)

    if alt == "two-sided":
        zu = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        zl = (-tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
    elif alt == "greater":
        zscore = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:
        zscore = (-tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
