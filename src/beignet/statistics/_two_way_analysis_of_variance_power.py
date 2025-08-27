import math

import torch
from torch import Tensor

import beignet.distributions


def two_way_analysis_of_variance_power(
    input: Tensor,
    sample_size_per_cell: Tensor,
    levels_factor_a: Tensor,
    levels_factor_b: Tensor,
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
    sample_size_per_cell : Tensor
        Sample size.
    levels_factor_a : Tensor
        Levels Factor A parameter.
    levels_factor_b : Tensor
        Levels Factor B parameter.
    alpha : float, default 0.05
        Type I error rate.
    effect_type : str, default 'main_a'
        Effect Type parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    sample_size_per_cell = torch.atleast_1d(torch.as_tensor(sample_size_per_cell))

    levels_factor_a = torch.atleast_1d(torch.as_tensor(levels_factor_a))
    levels_factor_b = torch.atleast_1d(torch.as_tensor(levels_factor_b))

    dtype = torch.float32
    for tensor in (input, sample_size_per_cell, levels_factor_a, levels_factor_b):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    sample_size_per_cell = sample_size_per_cell.to(dtype)

    levels_factor_a = levels_factor_a.to(dtype)
    levels_factor_b = levels_factor_b.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size_per_cell = torch.clamp(sample_size_per_cell, min=2.0)

    levels_factor_a = torch.clamp(levels_factor_a, min=2.0)
    levels_factor_b = torch.clamp(levels_factor_b, min=2.0)

    total_n = sample_size_per_cell * levels_factor_a * levels_factor_b

    if effect_type == "main_a":
        df_num = levels_factor_a - 1
    elif effect_type == "main_b":
        df_num = levels_factor_b - 1
    elif effect_type == "interaction":
        df_num = (levels_factor_a - 1) * (levels_factor_b - 1)
    else:
        raise ValueError("effect_type must be 'main_a', 'main_b', or 'interaction'")

    df_den = torch.clamp(
        levels_factor_a * levels_factor_b * (sample_size_per_cell - 1),
        min=1.0,
    )

    variance_f = (
        2
        * (df_num + 2 * total_n * input**2)
        / (df_num**2)
        * ((df_den + 2) / torch.clamp(df_den, min=1.0))
    )

    power = torch.clamp(
        0.5
        * (
            1
            - torch.erf(
                (
                    beignet.distributions.FisherSnedecor(df_num, df_den).icdf(
                        torch.tensor(1 - alpha, dtype=dtype),
                    )
                    - (df_num + total_n * input**2) / df_num
                )
                / torch.sqrt(torch.clamp(variance_f, min=1e-12))
                / math.sqrt(2.0),
            )
        ),
        0.0,
        1.0,
    )

    if out is not None:
        out.copy_(power)

        return out

    return power
