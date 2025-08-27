import math

import torch
from torch import Tensor

from ._independent_t_test_power import independent_t_test_power


def independent_t_test_sample_size(
    input: Tensor,
    ratio: Tensor | None = None,
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
    ratio : Tensor | None, optional
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
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    dtype = torch.float32
    for tensor in (input, ratio):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    ratio = ratio.to(dtype)

    input = torch.clamp(input, min=1e-6)

    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    variance_scaling = (1 + 1 / ratio) / 2

    sample_size_group_1_initial = ((z_alpha + z_beta) / input) ** 2 * variance_scaling

    sample_size_group_1_initial = torch.clamp(sample_size_group_1_initial, min=2.0)

    sample_size_group_1_iteration = sample_size_group_1_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        power_iteration = independent_t_test_power(
            input,
            sample_size_group_1_iteration,
            alpha,
            alternative,
            ratio,
        )

        power_iteration = torch.clamp(power_iteration, 0.01, 0.99)

        power_diff = power - power_iteration

        adjustment = (
            power_diff
            * sample_size_group_1_iteration
            / (2 * torch.clamp(power_iteration * (1 - power_iteration), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        sample_size_group_1_iteration = sample_size_group_1_iteration + adjustment

        sample_size_group_1_iteration = torch.clamp(
            sample_size_group_1_iteration,
            min=2.0,
            max=100000.0,
        )

    result = torch.ceil(sample_size_group_1_iteration)

    result = torch.clamp(result, min=2.0)

    if out is not None:
        out.copy_(result)
        return out
    return result
