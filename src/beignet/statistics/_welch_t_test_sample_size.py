import math

import torch
from torch import Tensor


def welch_t_test_sample_size(
    effect_size: Tensor,
    ratio: Tensor | float = 1.0,
    var_ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    r = torch.as_tensor(ratio)
    vr = torch.as_tensor(var_ratio)

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (effect_size, r, vr))
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    r = r.to(dtype) if isinstance(r, Tensor) else torch.tensor(float(r), dtype=dtype)
    vr = (
        vr.to(dtype) if isinstance(vr, Tensor) else torch.tensor(float(vr), dtype=dtype)
    )

    effect_size = torch.clamp(effect_size, min=1e-8)
    r = torch.clamp(r, min=0.1, max=10.0)
    vr = torch.clamp(vr, min=1e-6, max=1e6)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    variance_scaling_factor = 1.0 + vr / r
    sample_size_group_1_guess = (
        (z_alpha + z_beta) * torch.sqrt(variance_scaling_factor) / effect_size
    ) ** 2
    sample_size_group_1_guess = torch.clamp(sample_size_group_1_guess, min=2.0)

    sample_size_group_1_current = sample_size_group_1_guess
    max_iter = 12
    for _ in range(max_iter):
        sample_size_group_2_current = torch.clamp(
            torch.ceil(sample_size_group_1_current * r), min=2.0
        )
        a = 1.0 / sample_size_group_1_current
        b = vr / sample_size_group_2_current
        se2 = a + b
        se = torch.sqrt(se2)
        degrees_of_freedom = (se2**2) / (
            a**2 / torch.clamp(sample_size_group_1_current - 1, min=1.0)
            + b**2 / torch.clamp(sample_size_group_2_current - 1, min=1.0)
        )
        if alt == "two-sided":
            tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
        else:
            tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
        noncentrality_parameter = effect_size / torch.clamp(se, min=1e-12)
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
            p_curr = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            p_curr = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-tcrit - noncentrality_parameter) / torch.clamp(
                std_nct, min=1e-10
            )
            p_curr = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )

        gap = torch.clamp(power - p_curr, min=-0.49, max=0.49)
        sample_size_group_1_current = torch.clamp(
            sample_size_group_1_current * (1.0 + 1.25 * gap), min=2.0, max=1e7
        )

    result = torch.ceil(sample_size_group_1_current)
    result = torch.clamp(result, min=2.0)
    if out is not None:
        out.copy_(result)
        return out
    return result
