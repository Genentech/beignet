import math

import torch
from torch import Tensor

from ._two_one_sided_tests_two_sample_t_power import (
    two_one_sided_tests_two_sample_t_power,
)


def two_one_sided_tests_two_sample_t_sample_size(
    true_effect: Tensor,
    ratio: Tensor | float = 1.0,
    low: Tensor = 0.0,
    high: Tensor = 0.0,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    true_effect : Tensor
        True Effect parameter.
    ratio : Tensor | float, default 1.0
        Sample size ratio.
    low : Tensor, default 0.0
        Low parameter.
    high : Tensor, default 0.0
        High parameter.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    true_effect_size = torch.atleast_1d(torch.as_tensor(true_effect))

    r = torch.as_tensor(ratio)

    low = torch.atleast_1d(torch.as_tensor(low))

    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (true_effect_size, r, low, high))
        else torch.float32
    )
    true_effect_size = true_effect_size.to(dtype)

    low = low.to(dtype)

    high = high.to(dtype)

    r = r.to(dtype) if isinstance(r, Tensor) else torch.tensor(float(r), dtype=dtype)

    r = torch.clamp(r, min=0.1, max=10.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    margin = torch.minimum(true_effect_size - low, high - true_effect_size)

    margin = torch.clamp(margin, min=1e-8)

    c = torch.sqrt(1.0 + 1.0 / r)

    sample_size_group_1 = ((z_alpha + z_beta) * c / margin) ** 2

    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)

    sample_size_group_1_iteration = sample_size_group_1
    for _ in range(12):
        current_power = two_one_sided_tests_two_sample_t_power(
            true_effect_size,
            sample_size_group_1_iteration,
            ratio=r,
            low=low,
            high=high,
            alpha=alpha,
        )
        gap = torch.clamp(power - current_power, min=-0.45, max=0.45)

        sample_size_group_1_iteration = torch.clamp(
            sample_size_group_1_iteration * (1.0 + 1.25 * gap),
            min=2.0,
            max=1e7,
        )

    sample_size_group_1_output = torch.ceil(sample_size_group_1_iteration)

    if out is not None:
        out.copy_(sample_size_group_1_output)

        return out

    return sample_size_group_1_output
