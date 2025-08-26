import math

import torch
from torch import Tensor


def jonckheere_terpstra_test_power(
    input: Tensor,
    sample_sizes: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    sample_sizes : Tensor
        Sample size.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    sample_sizes = torch.atleast_1d(torch.as_tensor(sample_sizes))

    dtype = (
        torch.float64
        if (input.dtype == torch.float64 or sample_sizes.dtype == torch.float64)
        else torch.float32
    )
    input = input.to(dtype)

    sample_sizes = sample_sizes.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_sizes = torch.clamp(sample_sizes, min=2.0)

    groups = torch.tensor(sample_sizes.shape[-1], dtype=dtype)
    if groups < 3:
        raise ValueError("Jonckheere-Terpstra test requires at least 3 groups")

    n = torch.sum(sample_sizes, dim=-1)

    var_null = n * (n - 1) * (2 * n + 5) / 72

    std_null = torch.sqrt(torch.clamp(var_null, min=1e-12))

    mean_null = n * n / 4

    mean_alt = mean_null + input * std_null

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    critical = mean_null + z_alpha * std_null

    z_score = (critical - mean_alt) / std_null

    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
