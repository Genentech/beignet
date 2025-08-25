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
    """
    Minimum detectable absolute correlation |r| using Fisher z approximation.

    Parameters
    ----------
    sample_size : Tensor
        Total number of observations.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Test direction. This function returns a non-negative magnitude |r| for
        two-sided; for one-sided it returns the magnitude in the specified direction.

    Returns
    -------
    Tensor
        Minimal |r| (non-negative; apply sign per alternative if needed).
    """
    n0 = torch.as_tensor(sample_size)
    scalar_out = n0.ndim == 0
    n = torch.atleast_1d(n0)
    dtype = torch.float64 if n.dtype == torch.float64 else torch.float32
    n = torch.clamp(n.to(dtype), min=4.0)  # need n>3 for Fisher z

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Fisher z: SE = 1/sqrt(n-3); required |z_r| = (z_alpha+z_beta)*SE
    z_required = (z_alpha + z_beta) / torch.sqrt(torch.clamp(n - 3.0, min=1.0))
    # Inverse Fisher transform: r = tanh(z)
    r_mag = torch.tanh(torch.abs(z_required))

    if alt == "less":
        # return magnitude; user can apply sign if desired
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
