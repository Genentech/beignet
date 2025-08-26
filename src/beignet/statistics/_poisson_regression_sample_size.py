import math

import torch
from torch import Tensor

from ._poisson_regression_power import poisson_regression_power


def poisson_regression_sample_size(
    effect_size: Tensor,
    mean_rate: Tensor,
    p_exposure: Tensor = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    mean_rate = torch.atleast_1d(torch.as_tensor(mean_rate))
    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtypes = [effect_size.dtype, mean_rate.dtype, p_exposure.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    mean_rate = mean_rate.to(dtype)
    p_exposure = p_exposure.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.01, max=100.0)
    mean_rate = torch.clamp(mean_rate, min=0.01)
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

    mean_exposed = mean_rate * effect_size
    expected_count = p_exposure * mean_exposed + (1 - p_exposure) * mean_rate

    n_init = ((z_alpha + z_beta) ** 2) / (
        (beta**2) * p_exposure * (1 - p_exposure) * expected_count
    )
    n_init = torch.clamp(n_init, min=20.0)

    n_current = n_init
    for _ in range(15):
        current_power = poisson_regression_power(
            effect_size,
            n_current,
            mean_rate,
            p_exposure,
            alpha=alpha,
            alternative=alternative,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.2 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=20.0, max=1e6)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
