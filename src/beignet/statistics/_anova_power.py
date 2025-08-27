import torch
from torch import Tensor

import beignet.distributions


def anova_power(
    input: Tensor,
    sample_size: Tensor,
    groups: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    sample_size : Tensor
        Sample size.
    groups : Tensor
        Number of groups.
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
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    if (
        input.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or groups.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    groups = groups.to(dtype)

    input = torch.clamp(input, min=0.0)

    degrees_of_freedom_1 = groups - 1

    degrees_of_freedom_2 = sample_size - groups

    degrees_of_freedom_1 = torch.clamp(degrees_of_freedom_1, min=1.0)
    degrees_of_freedom_2 = torch.clamp(degrees_of_freedom_2, min=1.0)

    f_dist = beignet.distributions.FisherSnedecor(
        degrees_of_freedom_1, degrees_of_freedom_2
    )
    f_critical = f_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    lambda_nc = sample_size * input**2

    # Use non-central chi-squared distribution for the non-central F approximation
    nc_chi2_dist = beignet.distributions.NonCentralChi2(degrees_of_freedom_1, lambda_nc)

    # Get mean and variance from the distribution
    mean_nc_chi2 = nc_chi2_dist.mean
    variance_nc_chi2 = nc_chi2_dist.variance

    # Convert to F-distribution parameters
    mean_f = mean_nc_chi2 / degrees_of_freedom_1
    variance_f = variance_nc_chi2 / (degrees_of_freedom_1**2)

    adjustment = (degrees_of_freedom_2 + 2) / torch.clamp(degrees_of_freedom_2, min=1.0)
    variance_f = variance_f * adjustment

    standard_deviation_f = torch.sqrt(variance_f)

    z_score = (f_critical - mean_f) / torch.clamp(standard_deviation_f, min=1e-10)

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)
    power = 1 - normal_dist.cdf(z_score)

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out
    return result
