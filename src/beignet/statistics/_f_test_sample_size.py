import math

import torch
from torch import Tensor


def f_test_sample_size(
    input: Tensor,
    df1: Tensor,
    power: Tensor | float = 0.8,
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
    power : Tensor | float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    power = torch.atleast_1d(torch.as_tensor(power))

    df1 = torch.atleast_1d(torch.as_tensor(df1))

    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    if (
        input.dtype == torch.float64
        or df1.dtype == torch.float64
        or power.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    power = power.to(dtype)

    df1 = df1.to(dtype)

    power = torch.clamp(power, min=1e-6, max=1.0 - 1e-6)

    input = torch.clamp(input, min=1e-6)

    df1 = torch.clamp(df1, min=1.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(power) * square_root_two

    base_n = ((z_alpha + z_beta) / torch.sqrt(input)) ** 2

    n_estimate = base_n / df1 * (df1 + 2)

    min_n = df1 + 10

    max_n = torch.tensor(10000.0, dtype=dtype)

    result = torch.ceil(torch.clamp(n_estimate, min=min_n, max=max_n))

    if out is not None:
        out.copy_(result)
        return out
