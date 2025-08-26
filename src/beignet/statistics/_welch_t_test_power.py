import math

import torch
from torch import Tensor


def welch_t_test_power(
    effect_size: Tensor,
    nobs1: Tensor,
    nobs2: Tensor,
    var_ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    sample_size_group_2 = torch.atleast_1d(torch.as_tensor(nobs2))

    vr = torch.as_tensor(var_ratio)

    if any(
        t.dtype == torch.float64
        for t in (
            effect_size,
            sample_size_group_1,
            sample_size_group_2,
            vr if isinstance(vr, Tensor) else torch.tensor(0.0),
        )
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    sample_size_group_1 = sample_size_group_1.to(dtype)
    sample_size_group_2 = sample_size_group_2.to(dtype)
    if isinstance(vr, Tensor):
        vr = vr.to(dtype)
    else:
        vr = torch.tensor(float(vr), dtype=dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2, min=2.0)

    vr = torch.clamp(vr, min=1e-6, max=1e6)

    a = 1.0 / sample_size_group_1

    b = vr / sample_size_group_2

    se2 = a + b

    se = torch.sqrt(se2)

    degrees_of_freedom = (se2**2) / (
        a**2 / torch.clamp(sample_size_group_1 - 1, min=1.0)
        + b**2 / torch.clamp(sample_size_group_2 - 1, min=1.0)
    )

    ncp = effect_size / torch.clamp(se, min=1e-12)

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
        (degrees_of_freedom + ncp**2) / (degrees_of_freedom - 2),
        1 + ncp**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
    )
    std_nct = torch.sqrt(var_nct)

    if alt == "two-sided":
        zu = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)

        zl = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)

        power = 0.5 * (
            1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
    elif alt == "greater":
        zscore = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)

        power = 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:
        zscore = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)

        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    output = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(output)
        return out
    return output
