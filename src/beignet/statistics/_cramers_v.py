import torch
from torch import Tensor


def cramers_v(
    chi_square: Tensor,
    sample_size: Tensor,
    min_dim: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    chi_square : Tensor
        Chi Square parameter.
    sample_size : Tensor
        Sample size.
    min_dim : Tensor
        Min Dim parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Computed statistic.
    """

    chi_square = torch.atleast_1d(torch.as_tensor(chi_square))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    min_dim = torch.atleast_1d(torch.as_tensor(min_dim))

    dtype = chi_square.dtype
    if sample_size.dtype != dtype:
        if sample_size.dtype == torch.float64 or dtype == torch.float64:
            dtype = torch.float64
    if min_dim.dtype != dtype:
        if min_dim.dtype == torch.float64 or dtype == torch.float64:
            dtype = torch.float64

    chi_square = chi_square.to(dtype)

    sample_size = sample_size.to(dtype)

    min_dim = min_dim.to(dtype)

    result = torch.sqrt(chi_square / (sample_size * min_dim))

    result = torch.clamp(result, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out
