import math

import torch
from torch import Tensor


def correlation_minimum_detectable_effect(
    sample_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    sample_size : Tensor
        Sample size.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Effect size.
    """

    sample_size_0 = torch.as_tensor(sample_size)

    scalar_out = sample_size_0.ndim == 0

    sample_size = torch.atleast_1d(sample_size_0)

    dtype = torch.float64 if sample_size.dtype == torch.float64 else torch.float32

    sample_size = torch.clamp(sample_size.to(dtype), min=4.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    square_root_two = math.sqrt(2.0)
    if alt == "two-sided":
        z_alpha = (
            torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two
        )
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    z_required = (z_alpha + z_beta) / torch.sqrt(
        torch.clamp(sample_size - 3.0, min=1.0),
    )
    r_mag = torch.tanh(torch.abs(z_required))

    if alt == "less":
        out_t = r_mag
    else:
        out_t = r_mag

    if scalar_out:
        out_scalar = out_t.reshape(())
        if out is not None:
            out.copy_(out_scalar)
            return out
        return out_scalar
    else:
        if out is not None:
            out.copy_(out_t)
            return out
        return out_t
