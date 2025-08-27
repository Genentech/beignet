import math

import torch
from torch import Tensor

from ._two_one_sided_tests_one_sample_t_power import (
    two_one_sided_tests_one_sample_t_power,
)


def two_one_sided_tests_one_sample_t_sample_size(
    true_effect: Tensor,
    low: Tensor,
    high: Tensor,
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
    low : Tensor
        Low parameter.
    high : Tensor
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

    low = torch.atleast_1d(torch.as_tensor(low))

    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (true_effect_size, low, high))
        else torch.float32
    )
    true_effect_size = true_effect_size.to(dtype)

    low = low.to(dtype)

    high = high.to(dtype)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    margin = torch.minimum(true_effect_size - low, high - true_effect_size)

    margin = torch.clamp(margin, min=1e-8)

    n0 = ((z_alpha + z_beta) / margin) ** 2

    n0 = torch.clamp(n0, min=2.0)

    n_curr = n0
    for _ in range(12):
        current_power = two_one_sided_tests_one_sample_t_power(
            true_effect_size,
            n_curr,
            low,
            high,
            alpha=alpha,
        )
        gap = torch.clamp(power - current_power, min=-0.45, max=0.45)

        n_curr = torch.clamp(n_curr * (1.0 + 1.25 * gap), min=2.0, max=1e7)

    n_out = torch.ceil(n_curr)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
