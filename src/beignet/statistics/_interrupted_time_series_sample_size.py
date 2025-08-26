import math

import torch
from torch import Tensor

from ._interrupted_time_series_power import interrupted_time_series_power


def interrupted_time_series_sample_size(
    effect_size: Tensor,
    pre_post_ratio: Tensor = 1.0,
    autocorrelation: Tensor = 0.0,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    pre_post_ratio = torch.atleast_1d(torch.as_tensor(pre_post_ratio))

    autocorrelation = torch.atleast_1d(torch.as_tensor(autocorrelation))

    dtypes = [effect_size.dtype, pre_post_ratio.dtype, autocorrelation.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    pre_post_ratio = pre_post_ratio.to(dtype)

    autocorrelation = autocorrelation.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)

    pre_post_ratio = torch.clamp(pre_post_ratio, min=0.1, max=10.0)

    autocorrelation = torch.clamp(autocorrelation, min=-0.99, max=0.99)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    if torch.any(torch.abs(autocorrelation) > 1e-6):
        ar_adjustment = (1.0 - autocorrelation**2) / (1.0 + autocorrelation**2)
    else:
        ar_adjustment = torch.ones_like(autocorrelation)

    p_post = 1.0 / (1.0 + pre_post_ratio)

    design_variance = p_post * (1.0 - p_post)

    n_initial = ((z_alpha + z_beta) / effect_size) ** 2

    n_initial = n_initial / (ar_adjustment * design_variance)

    n_initial = torch.clamp(n_initial, min=10.0)

    n_pre_initial = torch.ceil(n_initial * pre_post_ratio / (1.0 + pre_post_ratio))

    n_pre_initial = torch.clamp(n_pre_initial, min=3.0)

    n_total_iteration = n_initial
    for _ in range(15):
        n_pre_iteration = torch.ceil(
            n_total_iteration * pre_post_ratio / (1.0 + pre_post_ratio),
        )
        n_pre_iteration = torch.clamp(
            n_pre_iteration,
            min=torch.tensor(3.0, dtype=dtype),
            max=n_total_iteration - 3.0,
        )

        current_power = interrupted_time_series_power(
            effect_size,
            n_total_iteration,
            n_pre_iteration,
            autocorrelation,
            alpha=alpha,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.4 * power_gap

        n_total_iteration = torch.clamp(
            n_total_iteration * adjustment,
            min=10.0,
            max=1e4,
        )

    n_out = torch.ceil(n_total_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
    return result
