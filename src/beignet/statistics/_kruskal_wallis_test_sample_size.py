import math

import torch
from torch import Tensor

from ._kruskal_wallis_test_power import kruskal_wallis_test_power


def kruskal_wallis_test_sample_size(
    input: Tensor,
    groups: Tensor | int,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    groups : Tensor | int
        Number of groups.
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

    input = torch.atleast_1d(torch.as_tensor(input))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    if input.dtype.is_floating_point and groups.dtype.is_floating_point:
        if input.dtype == torch.float64 or groups.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    input = input.to(dtype)

    groups = groups.to(dtype)

    input = torch.clamp(input, min=1e-8)

    groups = torch.clamp(groups, min=3.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    degrees_of_freedom = groups - 1

    n_initial = ((z_alpha + z_beta) ** 2) / (input * degrees_of_freedom)

    n_initial = torch.clamp(n_initial, min=5.0)

    n_iteration = n_initial
    for _ in range(12):
        sample_sizes = n_iteration.unsqueeze(-1).expand(
            *n_iteration.shape,
            int(groups.max().item()),
        )

        current_power = kruskal_wallis_test_power(
            input,
            sample_sizes,
            alpha=alpha,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=5.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
