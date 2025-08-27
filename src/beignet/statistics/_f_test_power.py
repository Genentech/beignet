import math

import torch
from torch import Tensor

import beignet.distributions


def f_test_power(
    input: Tensor,
    df1: Tensor,
    df2: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    df1 : Tensor
        Degrees of freedom.
    df2 : Tensor
        Degrees of freedom.
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

    df1 = torch.atleast_1d(torch.as_tensor(df1))
    df2 = torch.atleast_1d(torch.as_tensor(df2))

    if (
        input.dtype == torch.float64
        or df1.dtype == torch.float64
        or df2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    df1 = df1.to(dtype)
    df2 = df2.to(dtype)

    input = torch.clamp(input, min=0.0)

    df1 = torch.clamp(df1, min=1.0)
    df2 = torch.clamp(df2, min=1.0)

    sqrt_2 = math.sqrt(2.0)

    total_sample_size = df1 + df2 + 1

    lambda_param = total_sample_size * input

    f_dist = beignet.distributions.FisherSnedecor(df1, df2)
    f_critical = f_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    mean_f_alt = 1.0 + lambda_param / df1

    std_f_alt = torch.sqrt(2.0 / df1) * torch.sqrt(1.0 + 2.0 * lambda_param / df1)

    z_score = (f_critical - mean_f_alt) / torch.clamp(std_f_alt, min=1e-10)

    power = 0.5 * (1.0 - torch.erf(z_score / sqrt_2))

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out

    return result
