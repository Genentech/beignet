import math

import torch
from torch import Tensor

from ._proportional_hazards_model_power import proportional_hazards_model_power


def proportional_hazards_model_sample_size(
    hazard_ratio: Tensor,
    event_rate: Tensor,
    p_exposed: Tensor = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    hazard_ratio = torch.atleast_1d(torch.as_tensor(hazard_ratio))

    event_rate = torch.atleast_1d(torch.as_tensor(event_rate))

    p_exposed = torch.atleast_1d(torch.as_tensor(p_exposed))

    dtypes = [hazard_ratio.dtype, event_rate.dtype, p_exposed.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    hazard_ratio = hazard_ratio.to(dtype)

    event_rate = event_rate.to(dtype)

    p_exposed = p_exposed.to(dtype)

    hazard_ratio = torch.clamp(hazard_ratio, min=0.01, max=100.0)

    event_rate = torch.clamp(event_rate, min=0.01, max=0.99)

    p_exposed = torch.clamp(p_exposed, min=0.01, max=0.99)

    log_hr = torch.log(hazard_ratio)

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

    n_events_needed = ((z_alpha + z_beta) ** 2) / (
        p_exposed * (1.0 - p_exposed) * (log_hr**2)
    )
    n_events_needed = torch.clamp(n_events_needed, min=10.0)

    n_total_initial = n_events_needed / event_rate

    n_total_initial = torch.clamp(n_total_initial, min=20.0)

    n_iteration = n_total_initial
    for _ in range(10):
        expected_events = n_iteration * event_rate

        current_power = proportional_hazards_model_power(
            hazard_ratio,
            expected_events,
            p_exposed,
            alpha=alpha,
            alternative=alternative,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.1 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=20.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
    return result
