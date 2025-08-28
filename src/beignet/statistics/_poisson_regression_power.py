import functools

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

    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)

    mean_rate = torch.atleast_1d(mean_rate)

    p_exposure = torch.atleast_1d(p_exposure)

    dtype = functools.reduce(
        torch.promote_types,
        [input.dtype, sample_size.dtype, mean_rate.dtype, p_exposure.dtype],
    )

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    mean_rate = mean_rate.to(dtype)

    p_exposure = p_exposure.to(dtype)

    input = torch.clamp(input, min=0.01, max=100.0)

    sample_size = torch.clamp(sample_size, min=10.0)

    mean_rate = torch.clamp(mean_rate, min=0.01)

    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    standard_normal = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        power = (
            1
            - standard_normal.cdf(
                standard_normal.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
                - torch.abs(torch.log(input))
                / torch.sqrt(
                    torch.clamp(
                        1.0
                        / (
                            sample_size
                            * p_exposure
                            * (1 - p_exposure)
                            * (
                                p_exposure * mean_rate * input
                                + (1 - p_exposure) * mean_rate
                            )
                        ),
                        min=torch.finfo(dtype).eps,
                    ),
                ),
            )
        ) + standard_normal.cdf(
            -standard_normal.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
            - torch.abs(torch.log(input))
            / torch.sqrt(
                torch.clamp(
                    1.0
                    / (
                        sample_size
                        * p_exposure
                        * (1 - p_exposure)
                        * (
                            p_exposure * mean_rate * input
                            + (1 - p_exposure) * mean_rate
                        )
                    ),
                    min=torch.finfo(dtype).eps,
                ),
            ),
        )
    elif alternative == "greater":
        power = 1 - standard_normal.cdf(
            (
                standard_normal.icdf(torch.tensor(1 - alpha, dtype=dtype))
                - torch.abs(torch.log(input))
                / torch.sqrt(
                    torch.clamp(
                        1.0
                        / (
                            sample_size
                            * p_exposure
                            * (1 - p_exposure)
                            * (
                                p_exposure * mean_rate * input
                                + (1 - p_exposure) * mean_rate
                            )
                        ),
                        min=torch.finfo(dtype).eps,
                    ),
                )
            ),
        )
    else:
        power = standard_normal.cdf(
            -standard_normal.icdf(torch.tensor(1 - alpha, dtype=dtype))
            - torch.abs(torch.log(input))
            / torch.sqrt(
                torch.clamp(
                    1.0
                    / (
                        sample_size
                        * p_exposure
                        * (1 - p_exposure)
                        * (
                            p_exposure * mean_rate * input
                            + (1 - p_exposure) * mean_rate
                        )
                    ),
                    min=torch.finfo(dtype).eps,
                ),
            ),
        )

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)

        return out

    return power
