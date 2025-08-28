import torch
from torch import Tensor

import beignet.distributions


def chi_square_goodness_of_fit_power(
    input: Tensor,
    sample_size: Tensor,
    degrees_of_freedom: Tensor,
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
    degrees_of_freedom : Tensor
        Degrees of freedom.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)

    degrees_of_freedom = torch.atleast_1d(degrees_of_freedom)

    dtype = torch.promote_types(input.dtype, sample_size.dtype)
    dtype = torch.promote_types(dtype, degrees_of_freedom.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    degrees_of_freedom = degrees_of_freedom.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size = torch.clamp(sample_size, min=1.0)

    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    noncentrality = sample_size * input**2

    # Get critical value from central chi-squared distribution
    chi2_dist = beignet.distributions.Chi2(degrees_of_freedom)
    chi_squared_critical = chi2_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    # Use non-central chi-squared distribution for power calculation
    nc_chi2_dist = beignet.distributions.NonCentralChi2(
        degrees_of_freedom,
        noncentrality,
    )

    # Power is the probability that non-central chi-squared exceeds the critical value
    # P(X > critical) = 1 - CDF(critical) where X ~ NonCentralChi2(df, nc)
    power = 1 - nc_chi2_dist.cdf(chi_squared_critical)

    if out is not None:
        out.copy_(power)

        return out

    return power
