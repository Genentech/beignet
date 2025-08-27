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
        Input tensor.
    sample_size : Tensor
        Sample size.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    dtype = torch.float32
    for tensor in (input, sample_size):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)

    degrees_of_freedom = sample_size - 1

    noncentrality = input * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">", "one-sided", "one_sided"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Get critical values from central t-distribution
    t_dist = beignet.distributions.StudentT(degrees_of_freedom)
    if alt == "two-sided":
        t_critical = t_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        t_critical = t_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    # Use non-central t-distribution for power calculation
    nc_t_dist = beignet.distributions.NonCentralT(degrees_of_freedom, noncentrality)

    if alt == "two-sided":
        # P(|T| > t_critical) = P(T > t_critical) + P(T < -t_critical)
        power = (1 - nc_t_dist.cdf(t_critical)) + nc_t_dist.cdf(-t_critical)
    elif alt == "greater":
        # P(T > t_critical)
        power = 1 - nc_t_dist.cdf(t_critical)
    else:
        # P(T < -t_critical)
        power = nc_t_dist.cdf(-t_critical)

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out

    return result
