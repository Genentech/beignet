import math

import torch
from torch import Tensor


def chi_square_independence_power(
    effect_size: Tensor,
    sample_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    rows = torch.atleast_1d(torch.as_tensor(rows))
    cols = torch.atleast_1d(torch.as_tensor(cols))

    dtype = torch.float32
    for tensor in (effect_size, sample_size, rows, cols):
        dtype = torch.promote_types(dtype, tensor.dtype)

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    rows = rows.to(dtype)
    cols = cols.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    sample_size = torch.clamp(sample_size, min=1.0)

    rows = torch.clamp(rows, min=2.0)
    cols = torch.clamp(cols, min=2.0)

    degrees_of_freedom = (rows - 1) * (cols - 1)

    noncentrality = sample_size * effect_size**2

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
