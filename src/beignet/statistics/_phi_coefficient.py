import torch
from torch import Tensor


def phi_coefficient(
    chi_square: Tensor,
    sample_size: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    chi_square = torch.atleast_1d(torch.as_tensor(chi_square))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if chi_square.dtype != sample_size.dtype:
        if chi_square.dtype == torch.float64 or sample_size.dtype == torch.float64:
            chi_square = chi_square.to(torch.float64)

            sample_size = sample_size.to(torch.float64)
        else:
            chi_square = chi_square.to(torch.float32)

            sample_size = sample_size.to(torch.float32)

    output = torch.sqrt(chi_square / sample_size)

    output = torch.clamp(output, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
