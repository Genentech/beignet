import math

import torch
from torch import Tensor


def paired_t_test_sample_size(
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
    alternative : str, default 'two-sided'
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
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    sample_size = ((z_alpha + z_beta) / input) ** 2

    sample_size = torch.clamp(sample_size, min=2.0)

    sample_size_curr = sample_size
    for _ in range(10):
        degrees_of_freedom = torch.clamp(sample_size_curr - 1, min=1.0)

        t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

        noncentrality = input * torch.sqrt(sample_size_curr)

        variance_nct = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + noncentrality**2) / (degrees_of_freedom - 2),
            1 + noncentrality**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
        )
        standard_deviation_nct = torch.sqrt(variance_nct)
        if alt == "two-sided":
            zu = (t_critical - noncentrality) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            zl = (-t_critical - noncentrality) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            current_power = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (t_critical - noncentrality) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            current_power = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-t_critical - noncentrality) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )
            current_power = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        gap = torch.clamp(power - current_power, min=-0.45, max=0.45)

        sample_size_curr = torch.clamp(
            sample_size_curr * (1.0 + 1.25 * gap),
            min=2.0,
            max=1e7,
        )

    sample_size_out = torch.ceil(sample_size_curr)
    if out is not None:
        out.copy_(sample_size_out)
        return out
    return sample_size_out
