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
    r"""

    Parameters
    ----------
    hazard_ratio : Tensor
        Sample size ratio.
    n_events : Tensor
        N Events parameter.
    p_exposed : Tensor, default 0.5
        P Exposed parameter.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    hazard_ratio = torch.atleast_1d(hazard_ratio)

    n_events = torch.atleast_1d(n_events)

    p_exposed = torch.atleast_1d(p_exposed)

    dtype = torch.promote_types(hazard_ratio.dtype, n_events.dtype)
    dtype = torch.promote_types(dtype, p_exposed.dtype)

    hazard_ratio = hazard_ratio.to(dtype)

    n_events = n_events.to(dtype)

    p_exposed = p_exposed.to(dtype)

    hazard_ratio = torch.clamp(hazard_ratio, min=0.01, max=100.0)

    n_events = torch.clamp(n_events, min=5.0)

    p_exposed = torch.clamp(p_exposed, min=0.01, max=0.99)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    if alternative == "two-sided":
        power = 0.5 * (
            1
            - torch.erf(
                (
                    (
                        torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype))
                        * math.sqrt(2.0)
                    )
                    - torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                    * torch.abs(torch.log(hazard_ratio))
                )
                / math.sqrt(2.0),
            )
        ) + 0.5 * (
            1
            - torch.erf(
                (
                    (
                        torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype))
                        * math.sqrt(2.0)
                    )
                    + torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                    * torch.abs(torch.log(hazard_ratio))
                )
                / math.sqrt(2.0),
            )
        )
    elif alternative == "greater":
        if torch.all(hazard_ratio >= 1.0):
            power = 0.5 * (
                1
                - torch.erf(
                    (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                        * math.sqrt(2.0)
                        - torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                        * torch.abs(torch.log(hazard_ratio))
                    )
                    / math.sqrt(2.0),
                )
            )
        else:
            power = 0.5 * (
                1
                - torch.erf(
                    (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                        * math.sqrt(2.0)
                        - torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                        * torch.abs(torch.log(hazard_ratio))
                    )
                    / math.sqrt(2.0),
                )
            ) + 0.5 * (
                1
                - torch.erf(
                    (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                        * math.sqrt(2.0)
                        + torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                        * torch.abs(torch.log(hazard_ratio))
                    )
                    / math.sqrt(2.0),
                )
            )
    else:
        if torch.all(hazard_ratio <= 1.0):
            power = 0.5 * (
                1
                - torch.erf(
                    (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                        * math.sqrt(2.0)
                        + torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                        * torch.abs(torch.log(hazard_ratio))
                    )
                    / math.sqrt(2.0),
                )
            )
        else:
            power = 0.5 * (
                1
                - torch.erf(
                    (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                        * math.sqrt(2.0)
                        - torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                        * torch.abs(torch.log(hazard_ratio))
                    )
                    / math.sqrt(2.0),
                )
            ) + 0.5 * (
                1
                - torch.erf(
                    (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                        * math.sqrt(2.0)
                        + torch.sqrt(n_events * p_exposed * (1.0 - p_exposed))
                        * torch.abs(torch.log(hazard_ratio))
                    )
                    / math.sqrt(2.0),
                )
            )

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)

        return out

    return power
