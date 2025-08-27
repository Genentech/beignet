import math

import torch
from torch import Tensor

import beignet.distributions


def paired_z_test_sample_size(
    input: Tensor,
    power: float = 0.8,
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
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """
    input = torch.atleast_1d(torch.as_tensor(input))

    dtype = torch.float64 if input.dtype == torch.float64 else torch.float32

    input = torch.clamp(input.to(dtype), min=1e-8)

    square_root_two = math.sqrt(2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alt == "two-sided":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    z_beta = normal_dist.icdf(torch.tensor(power, dtype=dtype))

    sample_size = ((z_alpha + z_beta) / input) ** 2

    sample_size = torch.clamp(sample_size, min=1.0)

    sample_size = torch.ceil(sample_size)
    if out is not None:
        out.copy_(sample_size)
        return out
    return sample_size
