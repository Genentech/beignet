import math

import torch
from torch import Tensor

from ._jonckheere_terpstra_test_power import jonckheere_terpstra_test_power


def jonckheere_terpstra_test_sample_size(
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

    dtype = (
        torch.float64
        if (input.dtype == torch.float64 or groups.dtype == torch.float64)
        else torch.float32
    )
    input = input.to(dtype)

    groups = groups.to(dtype)

    input = torch.maximum(input, torch.finfo(dtype).eps)

    groups = torch.clamp(groups, min=3.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    n_initial = ((z_alpha + z_beta) / input) ** 2 / groups

    n_initial = torch.clamp(n_initial, min=5.0)

    n_iteration = n_initial
    for _ in range(12):
        sample_sizes = n_iteration.unsqueeze(-1).expand(
            *n_iteration.shape,
            int(groups.max().item()),
        )

        current_power = jonckheere_terpstra_test_power(
            input,
            sample_sizes,
            alpha=alpha,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.3 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=5.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
