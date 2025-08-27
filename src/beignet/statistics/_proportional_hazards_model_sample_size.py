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

    Parameters
    ----------
    hazard_ratio : Tensor
        Sample size ratio.
    event_rate : Tensor
        Event Rate parameter.
    p_exposed : Tensor, default 0.5
        P Exposed parameter.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
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

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * math.sqrt(
            2.0,
        )
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * math.sqrt(2.0)

    n_events_needed = torch.clamp(
        (
            (z_alpha + torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0))
            ** 2
        )
        / (p_exposed * (1.0 - p_exposed) * (torch.log(hazard_ratio) ** 2)),
        min=10.0,
    )

    n_iteration = torch.clamp(n_events_needed / event_rate, min=20.0)
    for _ in range(10):
        n_iteration = torch.clamp(
            n_iteration
            * (
                1.0
                + 1.1
                * torch.clamp(
                    power
                    - proportional_hazards_model_power(
                        hazard_ratio,
                        n_iteration * event_rate,
                        p_exposed,
                        alpha=alpha,
                        alternative=alternative,
                    ),
                    min=-0.4,
                    max=0.4,
                )
            ),
            min=20.0,
            max=1e6,
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
