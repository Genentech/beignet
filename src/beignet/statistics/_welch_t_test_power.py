import functools

import torch
from torch import Tensor

import beignet.distributions


def welch_t_test_power(
    input: Tensor,
    nobs1: Tensor,
    nobs2: Tensor,
    var_ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    nobs1 : Tensor

    nobs2 : Tensor

    var_ratio : Tensor | float, default 1.0

    alpha : float, default 0.05

    alternative : str, default 'two-sided'

    out : Tensor | None

    Returns
    -------
    Tensor
    """
    input = torch.atleast_1d(input)
    nobs1 = torch.atleast_1d(nobs1)
    nobs2 = torch.atleast_1d(nobs2)
    var_ratio = torch.atleast_1d(torch.as_tensor(var_ratio))

    dtype = functools.reduce(
        torch.promote_types,
        [input.dtype, nobs1.dtype, nobs2.dtype, var_ratio.dtype],
    )

    input = input.to(dtype)
    nobs1 = nobs1.to(dtype)
    nobs2 = nobs2.to(dtype)
    var_ratio = var_ratio.to(dtype)

    input = torch.clamp(input, min=0.0)
    nobs1 = torch.clamp(nobs1, min=2.0)
    nobs2 = torch.clamp(nobs2, min=2.0)
    var_ratio = torch.clamp(var_ratio, min=torch.finfo(dtype).eps, max=1e6)

    a = 1.0 / nobs1
    b = var_ratio / nobs2

    degrees_of_freedom = ((a + b) ** 2) / (
        a**2 / (nobs1 - 1)
        + b**2 / (nobs2 - 1)  # nobs1,nobs2 >= 2, so denominators >= 1
    )
    noncentrality = input / torch.clamp(torch.sqrt(a + b), min=torch.finfo(dtype).eps)

    t_dist = beignet.distributions.StudentT(degrees_of_freedom)
    if alternative == "two-sided":
        t_critical = t_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        t_critical = t_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    nc_t_dist = beignet.distributions.NonCentralT(degrees_of_freedom, noncentrality)

    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    std_dev = torch.sqrt(nc_t_dist.variance)  # sqrt is always positive

    if alternative == "two-sided":
        power = 0.5 * (
            1 - torch.erf((t_critical - nc_t_dist.mean) / std_dev / sqrt_2)
        ) + 0.5 * (1 + torch.erf((-t_critical - nc_t_dist.mean) / std_dev / sqrt_2))
    elif alternative == "greater":
        power = 0.5 * (1 - torch.erf((t_critical - nc_t_dist.mean) / std_dev / sqrt_2))
    else:
        power = 0.5 * (1 + torch.erf((-t_critical - nc_t_dist.mean) / std_dev / sqrt_2))

    # Power from erf operations is already bounded [0,1]
    output = power

    if out is not None:
        out.copy_(output)

        return out

    return output
