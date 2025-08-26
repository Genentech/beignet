import math

import torch
from torch import Tensor


def poisson_regression_power(
    effect_size: Tensor,
    sample_size: Tensor,
    mean_rate: Tensor,
    p_exposure: Tensor = 0.5,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    mean_rate = torch.atleast_1d(torch.as_tensor(mean_rate))

    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtypes = [effect_size.dtype, sample_size.dtype, mean_rate.dtype, p_exposure.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    mean_rate = mean_rate.to(dtype)

    p_exposure = p_exposure.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.01, max=100.0)

    sample_size = torch.clamp(sample_size, min=10.0)

    mean_rate = torch.clamp(mean_rate, min=0.01)

    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(effect_size)

    mean_unexposed = mean_rate

    mean_exposed = mean_rate * effect_size

    expected_count = p_exposure * mean_exposed + (1 - p_exposure) * mean_unexposed

    variance_beta = 1.0 / (sample_size * p_exposure * (1 - p_exposure) * expected_count)

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
    return result
