import math

import torch
from torch import Tensor

from ._logistic_regression_power import logistic_regression_power


def logistic_regression_sample_size(
    effect_size: Tensor,
    p_exposure: Tensor = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or p_exposure.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)

    p_exposure = p_exposure.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.01, max=100.0)

    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(effect_size)

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

    p_outcome_approximate = torch.tensor(0.5, dtype=dtype)

    variance_approximate = 1.0 / (
        p_exposure
        * (1 - p_exposure)
        * p_outcome_approximate
        * (1 - p_outcome_approximate)
    )

    n_initial = ((z_alpha + z_beta) ** 2) * variance_approximate / (beta**2)

    n_initial = torch.clamp(n_initial, min=20.0)

    n_iteration = n_initial
    for _ in range(15):
        current_power = logistic_regression_power(
            effect_size,
            n_iteration,
            p_exposure,
            alpha=alpha,
            alternative=alternative,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=20.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
