import functools

import torch
from torch import Tensor

import beignet.distributions


def t_test_power(
    input: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    sample_size : Tensor

    alpha : float, default 0.05

    alternative : str, default 'two-sided'

    out : Tensor | None

    Returns
    -------
    Tensor
    """
    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)

    dtype = functools.reduce(torch.promote_types, [input.dtype, sample_size.dtype])

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)

    degrees_of_freedom = sample_size - 1
    noncentrality = input * torch.sqrt(sample_size)

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

    if alternative == "two-sided":
        power = (1 - nc_t_dist.cdf(t_critical)) + nc_t_dist.cdf(-t_critical)
    elif alternative == "greater":
        power = 1 - nc_t_dist.cdf(t_critical)
    else:
        power = nc_t_dist.cdf(-t_critical)

    if out is not None:
        out.copy_(power)

        return out

    return power
