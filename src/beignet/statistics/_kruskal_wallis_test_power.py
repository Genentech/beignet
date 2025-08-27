import math

import torch
from torch import Tensor

import beignet.distributions


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

    # Get critical value from central chi-squared distribution
    chi2_dist = beignet.distributions.Chi2(degrees_of_freedom)
    chi_squared_critical = chi2_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    # Use non-central chi-squared distribution for power calculation
    nc_chi2_dist = beignet.distributions.NonCentralChi2(degrees_of_freedom, lambda_nc)

    # Get mean and variance from the distribution
    mean_nc_chi2 = nc_chi2_dist.mean
    variance_nc_chi2 = nc_chi2_dist.variance
    std_nc_chi2 = torch.sqrt(torch.clamp(variance_nc_chi2, min=1e-12))

    z_score = (chi_squared_critical - mean_nc_chi2) / std_nc_chi2

    square_root_two = math.sqrt(2.0)
    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
