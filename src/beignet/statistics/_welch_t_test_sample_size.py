import math

import torch
from torch import Tensor

from ._welch_t_test_power import welch_t_test_power


def welch_t_test_sample_size(
    input: Tensor,
    ratio: Tensor | float = 1.0,
    var_ratio: Tensor | float = 1.0,
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
    ratio : Tensor | float, default 1.0
        Sample size ratio.
    var_ratio : Tensor | float, default 1.0
        Sample size ratio.
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

    r = torch.as_tensor(ratio)

    vr = torch.as_tensor(var_ratio)

    dtype = torch.float32
    for tensor in (input, r, vr):
        dtype = torch.promote_types(dtype, tensor.dtype)
    input = input.to(dtype)

    r = r.to(dtype) if isinstance(r, Tensor) else torch.tensor(float(r), dtype=dtype)

    vr = (
        vr.to(dtype) if isinstance(vr, Tensor) else torch.tensor(float(vr), dtype=dtype)
    )

    input = torch.clamp(input, min=1e-8)

    r = torch.clamp(r, min=0.1, max=10.0)

    vr = torch.clamp(vr, min=1e-6, max=1e6)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    square_root_two = math.sqrt(2.0)
    if alt == "two-sided":
        z_alpha = (
            torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two
        )
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    variance_scaling_factor = 1.0 + vr / r

    sample_size_group_1_guess = (
        (z_alpha + z_beta) * torch.sqrt(variance_scaling_factor) / input
    ) ** 2
    sample_size_group_1_guess = torch.clamp(sample_size_group_1_guess, min=2.0)

    sample_size_group_1_iteration = sample_size_group_1_guess

    max_iter = 12
    for _ in range(max_iter):
        sample_size_group_2_iteration = torch.clamp(
            torch.ceil(sample_size_group_1_iteration * r),
            min=2.0,
        )

        p_curr = welch_t_test_power(
            input,
            sample_size_group_1_iteration,
            sample_size_group_2_iteration,
            vr,
            alpha,
            alternative,
        )

        gap = torch.clamp(power - p_curr, min=-0.49, max=0.49)

        sample_size_group_1_iteration = torch.clamp(
            sample_size_group_1_iteration * (1.0 + 1.25 * gap),
            min=2.0,
            max=1e7,
        )

    result = torch.ceil(sample_size_group_1_iteration)

    result = torch.clamp(result, min=2.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
