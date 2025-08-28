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
    r"""

    Parameters
    ----------
    kappa : Tensor
        Kappa parameter.
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

    kappa = torch.atleast_1d(kappa)

    sample_size = torch.atleast_1d(sample_size)

    dtype = torch.promote_types(kappa.dtype, sample_size.dtype)
    kappa = kappa.to(dtype)

    sample_size = sample_size.to(dtype)

    kappa = torch.clamp(kappa, min=-0.99, max=0.99)

    sample_size = torch.clamp(sample_size, min=10.0)

    p_e_approximate = torch.tensor(0.5, dtype=dtype)

    se_kappa = torch.sqrt(p_e_approximate / (sample_size * (1 - p_e_approximate)))

    noncentrality = torch.abs(kappa) / se_kappa

    sqrt2 = math.sqrt(2.0)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

        power = 0.5 * (1 - torch.erf((z_alpha - noncentrality) / sqrt2)) + 0.5 * (
            1 - torch.erf((z_alpha + noncentrality) / sqrt2)
        )
    elif alternative == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

        power = 0.5 * (1 - torch.erf((z_alpha - noncentrality) / sqrt2))
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

        power = 0.5 * (1 - torch.erf((z_alpha + noncentrality) / sqrt2))

    if out is not None:
        out.copy_(power)

        return out

    return power
