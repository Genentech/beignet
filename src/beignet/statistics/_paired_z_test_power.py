import torch
from torch import Tensor

import beignet.distributions


def paired_z_test_power(
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
    alternative : str, default "two-sided"
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

    dtype = (
        torch.float64
        if (input.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    input = input.to(dtype)

    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    noncentrality = input * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alt == "two-sided":
        upper = 1 - normal_dist.cdf(
            normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype)) - noncentrality,
        )
        lower = normal_dist.cdf(
            -normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype)) - noncentrality,
        )
        power = upper + lower
    elif alt == "greater":
        power = 1 - normal_dist.cdf(
            normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype)) - noncentrality,
        )
    else:
        power = normal_dist.cdf(
            -normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype)) - noncentrality,
        )

    out_t = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(out_t)

        return out

    return out_t
