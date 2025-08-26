import math

import torch
from torch import Tensor


def mcnemars_test_power(
    p01: Tensor,
    p10: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (p01, p10, sample_size))
        else torch.float32
    )
    p01 = torch.clamp(p01.to(dtype), 0.0, 1.0)
    p10 = torch.clamp(p10.to(dtype), 0.0, 1.0)

    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    d = sample_size * (p01 + p10)

    probability = torch.where(
        (p01 + p10) > 0, p01 / torch.clamp(p01 + p10, min=1e-12), torch.zeros_like(p01)
    )
    mean = d * (probability - 0.5)

    standard_deviation = torch.sqrt(torch.clamp(d * 0.25, min=1e-12))

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        pt = torch.tensor(prob, dtype=dtype)

        eps = torch.finfo(dtype).eps

        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if two_sided:
        z_critical = z_of(1 - alpha / 2)

        z_upper = z_critical - mean / torch.clamp(standard_deviation, min=1e-12)

        z_lower = -z_critical - mean / torch.clamp(standard_deviation, min=1e-12)

        power = 0.5 * (1 - torch.erf(z_upper / math.sqrt(2.0))) + 0.5 * (
            1 + torch.erf(z_lower / math.sqrt(2.0))
        )
    else:
        z_critical = z_of(1 - alpha)

        zscore = z_critical - mean / torch.clamp(standard_deviation, min=1e-12)

        power = 0.5 * (1 - torch.erf(zscore / math.sqrt(2.0)))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
