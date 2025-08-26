import math

import torch
from torch import Tensor


def kruskal_wallis_test_power(
    input: Tensor,
    sample_sizes: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    sample_sizes : Tensor
        Sample size.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    sample_sizes = torch.atleast_1d(torch.as_tensor(sample_sizes))

    if input.dtype.is_floating_point and sample_sizes.dtype.is_floating_point:
        if input.dtype == torch.float64 or sample_sizes.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    input = input.to(dtype)

    sample_sizes = sample_sizes.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_sizes = torch.clamp(sample_sizes, min=2.0)

    groups = torch.tensor(sample_sizes.shape[-1], dtype=dtype)

    n = torch.sum(sample_sizes, dim=-1)

    degrees_of_freedom = groups - 1

    lambda_nc = 12 * n * input / (n + 1)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    chi_squared_critical = degrees_of_freedom + z_alpha * torch.sqrt(
        2 * degrees_of_freedom,
    )

    mean_nc_chi2 = degrees_of_freedom + lambda_nc

    variance_nc_chi_squared = 2 * (degrees_of_freedom + 2 * lambda_nc)

    std_nc_chi2 = torch.sqrt(torch.clamp(variance_nc_chi_squared, min=1e-12))

    z_score = (chi_squared_critical - mean_nc_chi2) / std_nc_chi2

    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
