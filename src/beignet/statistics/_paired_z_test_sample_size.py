import math

import torch
from torch import Tensor


def paired_z_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    dtype = torch.float64 if effect_size.dtype == torch.float64 else torch.float32

    effect_size = torch.clamp(effect_size.to(dtype), min=1e-8)

    square_root_two = math.sqrt(2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    def z_of(p: float) -> torch.Tensor:
    r"""
    """
        pt = torch.tensor(p, dtype=dtype)

        eps = torch.finfo(dtype).eps

        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return square_root_two * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        z_alpha = z_of(1 - alpha / 2)
    else:
        z_alpha = z_of(1 - alpha)

    z_beta = z_of(power)

    sample_size = ((z_alpha + z_beta) / effect_size) ** 2

    sample_size = torch.clamp(sample_size, min=1.0)

    sample_size = torch.ceil(sample_size)
    if out is not None:
        out.copy_(sample_size)
        return out
    return sample_size
