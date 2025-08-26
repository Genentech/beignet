import math

import torch
from torch import Tensor


def interrupted_time_series_power(
    effect_size: Tensor,
    n_time_points: Tensor,
    n_pre_intervention: Tensor,
    autocorrelation: Tensor = 0.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    n_time_points = torch.atleast_1d(torch.as_tensor(n_time_points))

    n_pre_intervention = torch.atleast_1d(torch.as_tensor(n_pre_intervention))

    autocorrelation = torch.atleast_1d(torch.as_tensor(autocorrelation))

    dtypes = [
        effect_size.dtype,
        n_time_points.dtype,
        n_pre_intervention.dtype,
        autocorrelation.dtype,
    ]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    n_time_points = n_time_points.to(dtype)

    n_pre_intervention = n_pre_intervention.to(dtype)

    autocorrelation = autocorrelation.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    n_time_points = torch.clamp(n_time_points, min=6.0)

    n_pre_intervention = torch.clamp(
        n_pre_intervention,
        min=torch.tensor(3.0, dtype=dtype),
        max=n_time_points - 3.0,
    )
    autocorrelation = torch.clamp(autocorrelation, min=-0.99, max=0.99)

    n_post_intervention = n_time_points - n_pre_intervention

    if torch.any(torch.abs(autocorrelation) > 1e-6):
        ar_adjustment = (1.0 - autocorrelation**2) / (1.0 + autocorrelation**2)
    else:
        ar_adjustment = torch.ones_like(autocorrelation)

    effective_n = n_time_points * ar_adjustment

    prob_post = n_post_intervention / n_time_points
    design_variance = prob_post * (1.0 - prob_post)

    se_intervention = 1.0 / torch.sqrt(effective_n * design_variance)

    noncentrality = effect_size / se_intervention

    degrees_of_freedom_approximate = torch.clamp(effective_n - 4.0, min=1.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two

    t_critical = z_alpha * torch.sqrt(1.0 + 2.0 / degrees_of_freedom_approximate)

    z_score = t_critical - noncentrality
    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
