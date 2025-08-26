import math

import torch
from torch import Tensor


def proportional_hazards_model_power(
    hazard_ratio: Tensor,
    n_events: Tensor,
    p_exposed: Tensor = 0.5,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    hazard_ratio = torch.atleast_1d(torch.as_tensor(hazard_ratio))

    n_events = torch.atleast_1d(torch.as_tensor(n_events))

    p_exposed = torch.atleast_1d(torch.as_tensor(p_exposed))

    dtypes = [hazard_ratio.dtype, n_events.dtype, p_exposed.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    hazard_ratio = hazard_ratio.to(dtype)

    n_events = n_events.to(dtype)

    p_exposed = p_exposed.to(dtype)

    hazard_ratio = torch.clamp(hazard_ratio, min=0.01, max=100.0)

    n_events = torch.clamp(n_events, min=5.0)

    p_exposed = torch.clamp(p_exposed, min=0.01, max=0.99)

    log_hr = torch.log(hazard_ratio)

    variance_null = n_events * p_exposed * (1.0 - p_exposed)

    ncp = torch.sqrt(variance_null) * torch.abs(log_hr)

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

        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
            1 - torch.erf((z_alpha + ncp) / sqrt2)
        )
    elif alt == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        if torch.all(hazard_ratio >= 1.0):
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2))
        else:
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
                1 - torch.erf((z_alpha + ncp) / sqrt2)
            )
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        if torch.all(hazard_ratio <= 1.0):
            power = 0.5 * (1 - torch.erf((z_alpha + ncp) / sqrt2))
        else:
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
                1 - torch.erf((z_alpha + ncp) / sqrt2)
            )

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
