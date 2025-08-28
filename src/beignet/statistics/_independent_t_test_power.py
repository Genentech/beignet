import functools

import torch
from torch import Tensor

import beignet.distributions


def independent_t_test_power(
    input: Tensor,
    nobs1: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: Tensor | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    nobs1 : Tensor

    alpha : float, default 0.05

    alternative : str, default 'two-sided'

    ratio : Tensor | None, optional

    out : Tensor | None

    Returns
    -------
    Tensor
    """
    input = torch.atleast_1d(input)
    nobs1 = torch.atleast_1d(nobs1)

    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(ratio)

    dtype = functools.reduce(
        torch.promote_types,
        [input.dtype, nobs1.dtype, ratio.dtype],
    )

    input = input.to(dtype)
    nobs1 = nobs1.to(dtype)
    ratio = ratio.to(dtype)

    input = torch.clamp(input, min=0.0)
    nobs1 = torch.clamp(nobs1, min=2.0)
    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    nobs2 = nobs1 * ratio
    degrees_of_freedom = torch.clamp(nobs1 + nobs2 - 2, min=1.0)

    se_factor = torch.sqrt(1 / nobs1 + 1 / nobs2)
    noncentrality = input / se_factor

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    t_dist = beignet.distributions.StudentT(degrees_of_freedom)
    if alternative == "two-sided":
        t_critical = t_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        t_critical = t_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    nc_t_dist = beignet.distributions.NonCentralT(degrees_of_freedom, noncentrality)

    mean_nct = nc_t_dist.mean
    variance_nct = nc_t_dist.variance
    standard_deviation_nct = torch.sqrt(variance_nct)

    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))

    if alternative == "two-sided":
        z_upper = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=torch.finfo(dtype).eps,
        )
        z_lower = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=torch.finfo(dtype).eps,
        )

        power = 0.5 * (1 - torch.erf(z_upper / sqrt_2)) + 0.5 * (
            1 + torch.erf(z_lower / sqrt_2)
        )
    elif alternative == "greater":
        z_score = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=torch.finfo(dtype).eps,
        )
        power = 0.5 * (1 - torch.erf(z_score / sqrt_2))
    else:
        z_score = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=torch.finfo(dtype).eps,
        )
        power = 0.5 * (1 + torch.erf(z_score / sqrt_2))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
