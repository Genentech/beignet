import math

import torch
from torch import Tensor

from ._analysis_of_covariance_power import analysis_of_covariance_power


def analysis_of_covariance_sample_size(
    input: Tensor,
    groups: Tensor,
    covariate_r2: Tensor,
    n_covariates: Tensor | int = 1,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    groups : Tensor
        Number of groups.
    covariate_r2 : Tensor
        Covariate correlation.
    n_covariates : Tensor | int, default 1
        Covariate correlation.
    power : float, default 0.8
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

    effect_size_f = torch.atleast_1d(torch.as_tensor(input))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    covariate_r_squared = torch.atleast_1d(torch.as_tensor(covariate_r2))

    num_covariates = torch.atleast_1d(torch.as_tensor(n_covariates))

    dtype = torch.float32
    for tensor in (effect_size_f, groups, covariate_r_squared, num_covariates):
        dtype = torch.promote_types(dtype, tensor.dtype)
    effect_size_f = torch.clamp(effect_size_f.to(dtype), min=1e-8)

    groups = torch.clamp(groups.to(dtype), min=2.0)

    covariate_r_squared = torch.clamp(
        covariate_r_squared.to(dtype),
        min=0.0,
        max=1 - torch.finfo(dtype).eps,
    )

    num_covariates = torch.clamp(num_covariates.to(dtype), min=0.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    lam0 = ((z_alpha + z_beta) * math.sqrt(2.0)) ** 2

    n0 = (
        lam0
        * torch.clamp(1.0 - covariate_r_squared, min=torch.finfo(dtype).eps)
        / (effect_size_f**2)
    )

    n0 = torch.clamp(n0, min=groups + num_covariates + 2.0)

    n_curr = n0
    for _ in range(8):
        power_curr = analysis_of_covariance_power(
            effect_size_f,
            n_curr,
            groups,
            covariate_r_squared,
            num_covariates,
            alpha,
        )

        gap = torch.clamp(power - power_curr, min=-0.45, max=0.45)

        n_curr = torch.clamp(
            n_curr * (1.0 + 1.0 * gap),
            min=groups + num_covariates + 2.0,
            max=torch.tensor(1e7, dtype=dtype),
        )

    n_out = torch.ceil(n_curr)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
