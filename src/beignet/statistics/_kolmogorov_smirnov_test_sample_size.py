import math

import torch
from torch import Tensor

from ._kolmogorov_smirnov_test_power import kolmogorov_smirnov_test_power


def kolmogorov_smirnov_test_sample_size(
    input: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
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

    input = torch.atleast_1d(torch.as_tensor(input))

    dtype = input.dtype
    if not dtype.is_floating_point:
        dtype = torch.float32

    input = input.to(dtype)

    input = torch.clamp(input, min=1e-8, max=1.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alt == "two-sided":
        if alpha == 0.05:
            c_alpha = 1.36
        elif alpha == 0.01:
            c_alpha = 1.63
        else:
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha / 2, dtype=dtype)))
    else:
        if alpha == 0.05:
            c_alpha = 1.22
        elif alpha == 0.01:
            c_alpha = 1.52
        else:
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha, dtype=dtype)))

    square_root_two = math.sqrt(2.0)

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    n_initial = (c_alpha / input) ** 2

    n_initial = torch.clamp(n_initial, min=10.0)

    adjustment = 1.0 + 0.5 * (z_beta / c_alpha) ** 2

    n_initial = n_initial * adjustment

    n_iteration = n_initial
    for _ in range(12):
        current_power = kolmogorov_smirnov_test_power(
            input,
            n_iteration,
            alpha=alpha,
            alternative=alternative,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.5 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=10.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
