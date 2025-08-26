import math

import torch
from torch import Tensor


def paired_z_test_power(
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

    dtype = (
        torch.float64
        if (input.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    input = input.to(dtype)

    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    noncentrality = input * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    square_root_two = math.sqrt(2.0)

    def z_of(p: float) -> torch.Tensor:
        probability = torch.tensor(p, dtype=dtype)

        eps = torch.finfo(dtype).eps

        probability = torch.clamp(probability, min=eps, max=1 - eps)
        return square_root_two * torch.erfinv(2.0 * probability - 1.0)

    if alt == "two-sided":
        z_critical = z_of(1 - alpha / 2)

        upper = 0.5 * (
            1
            - torch.erf(
                (z_critical - noncentrality)
                / torch.sqrt(torch.tensor(2.0, dtype=dtype)),
            )
        )
        lower = 0.5 * (
            1
            + torch.erf(
                (-z_critical - noncentrality)
                / torch.sqrt(torch.tensor(2.0, dtype=dtype)),
            )
        )
        power = upper + lower
    elif alt == "greater":
        z_critical = z_of(1 - alpha)

        power = 0.5 * (
            1
            - torch.erf(
                (z_critical - noncentrality)
                / torch.sqrt(torch.tensor(2.0, dtype=dtype)),
            )
        )
    else:
        z_critical = z_of(1 - alpha)

        power = 0.5 * (
            1
            + torch.erf(
                (-z_critical - noncentrality)
                / torch.sqrt(torch.tensor(2.0, dtype=dtype)),
            )
        )

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
