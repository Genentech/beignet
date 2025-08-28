import math

import torch
from torch import Tensor


def logistic_regression_power(
    input: Tensor,
    sample_size: Tensor,
    p_exposure: Tensor | float = 0.5,
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

    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)

    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtype = torch.promote_types(input.dtype, sample_size.dtype)
    dtype = torch.promote_types(dtype, p_exposure.dtype)
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
    se_beta = torch.sqrt(torch.clamp(variance_beta, min=torch.finfo(dtype).eps))

    noncentrality = torch.abs(beta) / se_beta

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
