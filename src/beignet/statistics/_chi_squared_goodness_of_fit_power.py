import torch
from torch import Tensor


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

    input = torch.atleast_1d(torch.as_tensor(input))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    degrees_of_freedom = torch.atleast_1d(torch.as_tensor(degrees_of_freedom))

    dtype = torch.float32
    for tensor in (input, sample_size, degrees_of_freedom):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    degrees_of_freedom = degrees_of_freedom.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size = torch.clamp(sample_size, min=1.0)

    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    noncentrality = sample_size * input**2

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * torch.sqrt(
        torch.tensor(2.0, dtype=dtype),
    )
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

    power = 0.5 * (1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=dtype))))

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out
