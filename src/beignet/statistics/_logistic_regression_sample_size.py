import math

import torch
from torch import Tensor

from ._logistic_regression_power import logistic_regression_power


def logistic_regression_sample_size(
    input: Tensor,
    p_exposure: Tensor = 0.5,
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
    p_exposure : Tensor, default 0.5
        P Exposure parameter.
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

    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtype = (
        torch.float64
        if (input.dtype == torch.float64 or p_exposure.dtype == torch.float64)
        else torch.float32
    )
    input = input.to(dtype)

    p_exposure = p_exposure.to(dtype)

    input = torch.clamp(input, min=0.01, max=100.0)

    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(input)

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

    p_outcome_approximate = torch.tensor(0.5, dtype=dtype)

    variance_approximate = 1.0 / (
        p_exposure
        * (1 - p_exposure)
        * p_outcome_approximate
        * (1 - p_outcome_approximate)
    )

    n_initial = ((z_alpha + z_beta) ** 2) * variance_approximate / (beta**2)

    n_initial = torch.clamp(n_initial, min=20.0)

    n_iteration = n_initial
    for _ in range(15):
        current_power = logistic_regression_power(
            input,
            n_iteration,
            p_exposure,
            alpha=alpha,
            alternative=alternative,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=20.0, max=1e6)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
