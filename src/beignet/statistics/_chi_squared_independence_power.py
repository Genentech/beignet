import math

import torch
from torch import Tensor


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

    square_root_2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_2

    chi_squared_critical = degrees_of_freedom + z_alpha * torch.sqrt(
        2 * degrees_of_freedom,
    )

    mean_nc_chi2 = degrees_of_freedom + noncentrality

    variance_nc_chi_squared = 2 * (degrees_of_freedom + 2 * noncentrality)

    std_nc_chi2 = torch.sqrt(variance_nc_chi_squared)

    z_score = (chi_squared_critical - mean_nc_chi2) / torch.clamp(
        std_nc_chi2,
        min=1e-10,
    )

    result = torch.clamp((1 - torch.erf(z_score / square_root_2)) / 2, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out

    return result
