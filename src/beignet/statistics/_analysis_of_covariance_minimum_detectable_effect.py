import math

import torch
from torch import Tensor

from ._analysis_of_covariance_power import analysis_of_covariance_power


def analysis_of_covariance_minimum_detectable_effect(
    sample_size: Tensor,
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
    sample_size : Tensor
        Sample size.
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
        Effect size.
    """

    sample_size_tensor = torch.as_tensor(sample_size)

    groups_tensor = torch.as_tensor(groups)

    covariate_r_squared_tensor = torch.as_tensor(covariate_r2)

    num_covariates_tensor = torch.as_tensor(n_covariates)

    sample_size_1d = torch.atleast_1d(sample_size_tensor)

    groups_1d = torch.atleast_1d(groups_tensor)

    covariate_r_squared_1d = torch.atleast_1d(covariate_r_squared_tensor)

    num_covariates_1d = torch.atleast_1d(num_covariates_tensor)

    dtype = torch.float32
    for tensor in (
        sample_size_1d,
        groups_1d,
        covariate_r_squared_1d,
        num_covariates_1d,
    ):
        dtype = torch.promote_types(dtype, tensor.dtype)

    sample_size_clamped = torch.clamp(sample_size_1d.to(dtype), min=3.0)

    groups_clamped = torch.clamp(groups_1d.to(dtype), min=2.0)

    covariate_r_squared_clamped = torch.clamp(
        covariate_r_squared_1d.to(dtype),
        min=0.0,
        max=1 - torch.finfo(dtype).eps,
    )

    num_covariates_clamped = torch.clamp(num_covariates_1d.to(dtype), min=0.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    degrees_of_freedom_1 = torch.clamp(groups_clamped - 1.0, min=1.0)

    square_root_degrees_freedom_over_sample_size = torch.sqrt(
        degrees_of_freedom_1 / torch.clamp(sample_size_clamped, min=1.0),
    )

    square_root_residual_variance = torch.sqrt(
        torch.clamp(1.0 - covariate_r_squared_clamped, min=torch.finfo(dtype).eps),
    )

    starting_effect_size = torch.clamp(
        (z_alpha + z_beta)
        * square_root_degrees_freedom_over_sample_size
        * square_root_residual_variance,
        min=1e-8,
    )

    minimum_effect_size = 1e-8

    maximum_effect_size_epsilon = 1e-6

    effect_size_bottom = torch.zeros_like(starting_effect_size) + minimum_effect_size

    effect_size_top = torch.clamp(
        2.0 * starting_effect_size + maximum_effect_size_epsilon,
        min=maximum_effect_size_epsilon,
    )

    maximum_expansion_iterations = 8

    for _ in range(maximum_expansion_iterations):
        power_target = analysis_of_covariance_power(
            effect_size_top,
            sample_size_clamped,
            groups_clamped,
            covariate_r_squared_clamped,
            num_covariates_clamped,
            alpha,
        )

        needs_expansion = power_target < power

        if not torch.any(needs_expansion):
            break

        effect_size_top = torch.where(
            needs_expansion,
            effect_size_top * 2.0,
            effect_size_top,
        )

        effect_size_top = torch.clamp(
            effect_size_top,
            max=torch.tensor(10.0, dtype=dtype),
        )

    maximum_bisection_iterations = 24

    effect_size_center = (effect_size_bottom + effect_size_top) * 0.5

    for _ in range(maximum_bisection_iterations):
        power_center = analysis_of_covariance_power(
            effect_size_center,
            sample_size_clamped,
            groups_clamped,
            covariate_r_squared_clamped,
            num_covariates_clamped,
            alpha,
        )

        power_below_target = power_center < power

        effect_size_bottom = torch.where(
            power_below_target,
            effect_size_center,
            effect_size_bottom,
        )

        effect_size_top = torch.where(
            power_below_target,
            effect_size_top,
            effect_size_center,
        )

        effect_size_center = (effect_size_bottom + effect_size_top) * 0.5

    output = torch.clamp(effect_size_center, min=0.0)

    if out is not None:
        out.copy_(output)
        return out
    return output
