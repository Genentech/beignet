import math

import torch
from torch import Tensor

import beignet.distributions


def multivariate_analysis_of_variance_power(
    input: Tensor,
    sample_size: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
    alpha: float = 0.05,
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
    n_variables : Tensor
        N Variables parameter.
    n_groups : Tensor
        Number of groups.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)
    n_variables = torch.atleast_1d(n_variables)

    n_groups = torch.atleast_1d(n_groups)

    dtype = torch.promote_types(input.dtype, sample_size.dtype)
    dtype = torch.promote_types(dtype, n_variables.dtype)
    dtype = torch.promote_types(dtype, n_groups.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)
    n_variables = n_variables.to(dtype)

    n_groups = n_groups.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size = torch.clamp(sample_size, min=n_groups + n_variables + 5)

    n_variables = torch.clamp(n_variables, min=1.0)

    n_groups = torch.clamp(n_groups, min=2.0)

    power = torch.clamp(
        0.5
        * (
            1
            - torch.erf(
                (
                    beignet.distributions.FisherSnedecor(
                        (n_groups - 1) * n_variables,
                        torch.clamp(
                            (sample_size - n_groups) * n_variables
                            - (n_variables - (n_groups - 1) + 1) / 2,
                            min=1.0,
                        ),
                    ).icdf(
                        torch.tensor(1 - alpha, dtype=dtype),
                    )
                    - (1.0 + sample_size * input**2 / ((n_groups - 1) * n_variables))
                    * (
                        torch.clamp(
                            (sample_size - n_groups) * n_variables
                            - (n_variables - (n_groups - 1) + 1) / 2,
                            min=1.0,
                        )
                        / (
                            torch.clamp(
                                (sample_size - n_groups) * n_variables
                                - (n_variables - (n_groups - 1) + 1) / 2,
                                min=1.0,
                            )
                            - 2.0
                        )
                    )
                )
                / torch.sqrt(
                    torch.clamp(
                        (
                            2.0
                            * (
                                torch.clamp(
                                    (sample_size - n_groups) * n_variables
                                    - (n_variables - (n_groups - 1) + 1) / 2,
                                    min=1.0,
                                )
                                / (
                                    torch.clamp(
                                        (sample_size - n_groups) * n_variables
                                        - (n_variables - (n_groups - 1) + 1) / 2,
                                        min=1.0,
                                    )
                                    - 2.0
                                )
                            )
                            ** 2
                            * (
                                ((n_groups - 1) * n_variables + sample_size * input**2)
                                / ((n_groups - 1) * n_variables)
                                + (
                                    torch.clamp(
                                        (sample_size - n_groups) * n_variables
                                        - (n_variables - (n_groups - 1) + 1) / 2,
                                        min=1.0,
                                    )
                                    - 2.0
                                )
                                / torch.clamp(
                                    (sample_size - n_groups) * n_variables
                                    - (n_variables - (n_groups - 1) + 1) / 2,
                                    min=1.0,
                                )
                            )
                        ),
                        min=1e-12,
                    ),
                )
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
