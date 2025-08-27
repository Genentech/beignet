import torch
from torch import Tensor

import beignet.distributions


def correlation_power(
    r: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    r : Tensor
        Correlation coefficient.
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

    r = torch.atleast_1d(torch.as_tensor(r))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if r.dtype != sample_size.dtype:
        if r.dtype == torch.float64 or sample_size.dtype == torch.float64:
            r = r.to(torch.float64)

            sample_size = sample_size.to(torch.float64)
        else:
            r = r.to(torch.float32)

            sample_size = sample_size.to(torch.float32)

    epsilon = 1e-7

    r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)

    z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

    se_z = 1.0 / torch.sqrt(sample_size - 3)

    z_stat = z_r / se_z

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError(f"Unknown alternative: {alternative}")

    normal_dist = beignet.distributions.StandardNormal.from_dtype(r.dtype)

    if alt == "two-sided":
        z_alpha_2 = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=r.dtype))

        cdf_upper = normal_dist.cdf(z_alpha_2 - z_stat)
        cdf_lower = normal_dist.cdf(-z_alpha_2 - z_stat)
        power = 1 - (cdf_upper - cdf_lower)

    elif alt == "greater":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=r.dtype))

        power = 1 - normal_dist.cdf(z_alpha - z_stat)

    elif alt == "less":
        z_alpha = normal_dist.icdf(torch.tensor(alpha, dtype=r.dtype))

        power = normal_dist.cdf(z_alpha - z_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
