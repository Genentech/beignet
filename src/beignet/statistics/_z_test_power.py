import torch
from torch import Tensor

import beignet.distributions


def z_test_power(
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

    dtype = torch.float32
    for tensor in (input, sample_size):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=1.0)

    noncentrality = input * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alt == "two-sided":
        z_alpha_half = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        power_upper = 1 - normal_dist.cdf(z_alpha_half - noncentrality)
        power_lower = normal_dist.cdf(-z_alpha_half - noncentrality)
        power = power_upper + power_lower
    elif alt == "greater":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = 1 - normal_dist.cdf(z_alpha - noncentrality)
    else:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = normal_dist.cdf(-z_alpha - noncentrality)

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
