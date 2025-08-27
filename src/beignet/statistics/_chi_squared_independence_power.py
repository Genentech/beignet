import torch
from torch import Tensor

import beignet.distributions


def chi_square_independence_power(
    input: Tensor,
    sample_size: Tensor,
    rows: Tensor,
    cols: Tensor,
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
    rows : Tensor
        Rows parameter.
    cols : Tensor
        Cols parameter.
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

    rows = torch.atleast_1d(torch.as_tensor(rows))
    cols = torch.atleast_1d(torch.as_tensor(cols))

    dtype = torch.float32
    for tensor in (input, sample_size, rows, cols):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    rows = rows.to(dtype)
    cols = cols.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size = torch.clamp(sample_size, min=1.0)

    rows = torch.clamp(rows, min=2.0)
    cols = torch.clamp(cols, min=2.0)

    degrees_of_freedom = (rows - 1) * (cols - 1)

    noncentrality = sample_size * input**2

    # Get critical value from central chi-squared distribution
    chi2_dist = beignet.distributions.Chi2(degrees_of_freedom)
    chi_squared_critical = chi2_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    # Use non-central chi-squared distribution for power calculation
    nc_chi2_dist = beignet.distributions.NonCentralChi2(
        degrees_of_freedom, noncentrality
    )

    # Power is the probability that non-central chi-squared exceeds the critical value
    power = 1 - nc_chi2_dist.cdf(chi_squared_critical)

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out

    return result
