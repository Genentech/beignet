import math

import torch
from torch import Tensor

from ._two_way_analysis_of_variance_power import two_way_analysis_of_variance_power


def two_way_analysis_of_variance_sample_size(
    input: Tensor,
    levels_factor_a: Tensor,
    levels_factor_b: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    effect_type: str = "main_a",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    levels_factor_a : Tensor
        Levels Factor A parameter.
    levels_factor_b : Tensor
        Levels Factor B parameter.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    effect_type : str, default 'main_a'
        Effect Type parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    levels_factor_a = torch.atleast_1d(torch.as_tensor(levels_factor_a))
    levels_factor_b = torch.atleast_1d(torch.as_tensor(levels_factor_b))

    dtypes = [input.dtype, levels_factor_a.dtype, levels_factor_b.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    levels_factor_a = levels_factor_a.to(dtype)
    levels_factor_b = levels_factor_b.to(dtype)

    input = torch.clamp(input, min=1e-8)

    levels_factor_a = torch.clamp(levels_factor_a, min=2.0)
    levels_factor_b = torch.clamp(levels_factor_b, min=2.0)

    if effect_type == "main_a":
        df_effect = levels_factor_a - 1
    elif effect_type == "main_b":
        df_effect = levels_factor_b - 1
    elif effect_type == "interaction":
        df_effect = (levels_factor_a - 1) * (levels_factor_b - 1)
    else:
        raise ValueError("effect_type must be 'main_a', 'main_b', or 'interaction'")

    n_iteration = torch.clamp(
        (
            (
                torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * math.sqrt(2.0)
                + torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
            )
            * torch.sqrt(df_effect)
        )
        ** 2
        / (input**2 * levels_factor_a * levels_factor_b),
        min=5.0,
    )
    for _ in range(12):
        n_iteration = torch.clamp(
            n_iteration
            * (
                1.0
                + 1.1
                * torch.clamp(
                    power
                    - two_way_analysis_of_variance_power(
                        input,
                        n_iteration,
                        levels_factor_a,
                        levels_factor_b,
                        alpha=alpha,
                        effect_type=effect_type,
                    ),
                    min=-0.4,
                    max=0.4,
                )
            ),
            min=5.0,
            max=1e5,
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
