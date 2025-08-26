import torch
from torch import Tensor


def chi_square_goodness_of_fit_power(
    effect_size: Tensor,
    sample_size: Tensor,
    df: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    degrees_of_freedom = torch.atleast_1d(torch.as_tensor(df))

    if (
        effect_size.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or degrees_of_freedom.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    degrees_of_freedom = degrees_of_freedom.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    sample_size = torch.clamp(sample_size, min=1.0)
    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    noncentrality_parameter = sample_size * effect_size**2

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * torch.sqrt(
        torch.tensor(2.0, dtype=dtype)
    )
    chi2_critical = degrees_of_freedom + z_alpha * torch.sqrt(2 * degrees_of_freedom)

    mean_nc_chi2 = degrees_of_freedom + noncentrality_parameter
    var_nc_chi2 = 2 * (degrees_of_freedom + 2 * noncentrality_parameter)
    std_nc_chi2 = torch.sqrt(var_nc_chi2)

    z_score = (chi2_critical - mean_nc_chi2) / torch.clamp(std_nc_chi2, min=1e-10)

    power = 0.5 * (1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=dtype))))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
