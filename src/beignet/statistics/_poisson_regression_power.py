import torch
from torch import Tensor

import beignet.distributions


def poisson_regression_power(
    input: Tensor,
    sample_size: Tensor,
    mean_rate: Tensor,
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
    mean_rate : Tensor
        Mean Rate parameter.
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

    mean_rate = torch.atleast_1d(torch.as_tensor(mean_rate))

    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtypes = [input.dtype, sample_size.dtype, mean_rate.dtype, p_exposure.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    mean_rate = mean_rate.to(dtype)

    p_exposure = p_exposure.to(dtype)

    input = torch.clamp(input, min=0.01, max=100.0)

    sample_size = torch.clamp(sample_size, min=10.0)

    mean_rate = torch.clamp(mean_rate, min=0.01)

    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(input)

    mean_unexposed = mean_rate

    mean_exposed = mean_rate * input

    expected_count = p_exposure * mean_exposed + (1 - p_exposure) * mean_unexposed

    variance_beta = 1.0 / (sample_size * p_exposure * (1 - p_exposure) * expected_count)

    se_beta = torch.sqrt(torch.clamp(variance_beta, min=1e-12))

    noncentrality = torch.abs(beta) / se_beta

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Use standard normal distribution for critical values
    standard_normal = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alt == "two-sided":
        z_alpha = standard_normal.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        zu = z_alpha - noncentrality
        zl = -z_alpha - noncentrality

        power = (1 - standard_normal.cdf(zu)) + standard_normal.cdf(zl)
    elif alt == "greater":
        z_alpha = standard_normal.icdf(torch.tensor(1 - alpha, dtype=dtype))

        z_score = z_alpha - noncentrality
        power = 1 - standard_normal.cdf(z_score)
    else:
        z_alpha = standard_normal.icdf(torch.tensor(1 - alpha, dtype=dtype))

        z_score = -z_alpha - noncentrality
        power = standard_normal.cdf(z_score)

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)

        return out

    return power
