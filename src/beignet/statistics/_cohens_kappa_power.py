import math

import torch
from torch import Tensor


def cohens_kappa_power(
    kappa: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    kappa = torch.atleast_1d(torch.as_tensor(kappa))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    dtype = (
        torch.float64
        if (kappa.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    kappa = kappa.to(dtype)

    sample_size = sample_size.to(dtype)

    kappa = torch.clamp(kappa, min=-0.99, max=0.99)

    sample_size = torch.clamp(sample_size, min=10.0)

    p_e_approximate = torch.tensor(0.5, dtype=dtype)

    se_kappa = torch.sqrt(p_e_approximate / (sample_size * (1 - p_e_approximate)))

    noncentrality = torch.abs(kappa) / se_kappa

    sqrt2 = math.sqrt(2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

        power = 0.5 * (1 - torch.erf((z_alpha - noncentrality) / sqrt2)) + 0.5 * (
            1 - torch.erf((z_alpha + noncentrality) / sqrt2)
        )
    elif alt == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

        power = 0.5 * (1 - torch.erf((z_alpha - noncentrality) / sqrt2))
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

        power = 0.5 * (1 - torch.erf((z_alpha + noncentrality) / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
    return result
