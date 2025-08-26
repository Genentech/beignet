import math

import torch
from torch import Tensor


def logistic_regression_power(
    input: Tensor,
    sample_size: Tensor,
    p_exposure: Tensor = 0.5,
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
    p_exposure : Tensor, default 0.5
        P Exposure parameter.
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

    input = torch.atleast_1d(torch.as_tensor(input))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (input, sample_size, p_exposure))
        else torch.float32
    )
    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    p_exposure = p_exposure.to(dtype)

    input = torch.clamp(input, min=0.01, max=100.0)

    sample_size = torch.clamp(sample_size, min=10.0)

    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(input)

    logit_baseline = torch.tensor(0.0, dtype=dtype)

    logit_exposed = logit_baseline + beta

    logit_unexposed = logit_baseline

    p_outcome_exposed = torch.sigmoid(logit_exposed)

    p_outcome_unexposed = torch.sigmoid(logit_unexposed)

    p_outcome = p_exposure * p_outcome_exposed + (1 - p_exposure) * p_outcome_unexposed

    p_outcome = torch.clamp(p_outcome, min=0.01, max=0.99)

    variance_beta = 1.0 / (
        sample_size * p_exposure * (1 - p_exposure) * p_outcome * (1 - p_outcome)
    )
    se_beta = torch.sqrt(torch.clamp(variance_beta, min=1e-12))

    noncentrality = torch.abs(beta) / se_beta

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
