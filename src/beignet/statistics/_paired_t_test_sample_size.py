import math

import torch
from torch import Tensor

from ._paired_t_test_power import paired_t_test_power


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

    input = torch.clamp(input.to(dtype), min=torch.finfo(dtype).eps)

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

    sample_size_curr = torch.clamp(
        ((z_alpha + torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2) / input)
        ** 2,
        min=2.0,
    )
    for _ in range(10):
        sample_size_curr = torch.clamp(
            sample_size_curr
            * (
                1.0
                + 1.25
                * torch.clamp(
                    power
                    - paired_t_test_power(
                        input,
                        sample_size_curr,
                        alpha,
                        alternative,
                    ),
                    min=-0.45,
                    max=0.45,
                )
            ),
            min=2.0,
            max=1e7,
        )

    sample_size_out = torch.ceil(sample_size_curr)

    if out is not None:
        out.copy_(sample_size_out)

        return out

    return sample_size_out
