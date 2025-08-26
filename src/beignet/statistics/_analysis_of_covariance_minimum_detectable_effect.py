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
    sample_size_tensor = torch.as_tensor(sample_size)

    groups_tensor = torch.as_tensor(groups)

    covariate_r2_tensor = torch.as_tensor(covariate_r2)

    num_covariates_tensor = torch.as_tensor(n_covariates)

    scalar_output = (
        sample_size_tensor.ndim == 0
        and groups_tensor.ndim == 0
        and covariate_r2_tensor.ndim == 0
        and num_covariates_tensor.ndim == 0
    )

    sample_size_1d = torch.atleast_1d(sample_size_tensor)

    groups_1d = torch.atleast_1d(groups_tensor)

    covariate_r2_1d = torch.atleast_1d(covariate_r2_tensor)

    num_covariates_1d = torch.atleast_1d(num_covariates_tensor)

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64
            for t in (sample_size_1d, groups_1d, covariate_r2_1d, num_covariates_1d)
        )
        else torch.float32
    )

    sample_size_clamped = torch.clamp(sample_size_1d.to(dtype), min=3.0)

    groups_clamped = torch.clamp(groups_1d.to(dtype), min=2.0)

    covariate_r2_clamped = torch.clamp(
        covariate_r2_1d.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps
    )

    num_covariates_clamped = torch.clamp(num_covariates_1d.to(dtype), min=0.0)

    sqrt_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_two

    degrees_of_freedom_1 = torch.clamp(groups_clamped - 1.0, min=1.0)

    sqrt_df_over_n = torch.sqrt(
        degrees_of_freedom_1 / torch.clamp(sample_size_clamped, min=1.0)
    )

    sqrt_residual_variance = torch.sqrt(
        torch.clamp(1.0 - covariate_r2_clamped, min=torch.finfo(dtype).eps)
    )

    initial_effect_size = torch.clamp(
        (z_alpha + z_beta) * sqrt_df_over_n * sqrt_residual_variance,
        min=1e-8,
    )

    minimum_effect_size = 1e-8

    maximum_effect_size_epsilon = 1e-6

    effect_size_lower = torch.zeros_like(initial_effect_size) + minimum_effect_size

    effect_size_upper = torch.clamp(
        2.0 * initial_effect_size + maximum_effect_size_epsilon,
        min=maximum_effect_size_epsilon,
    )

    max_expansion_iterations = 8

    for _ in range(max_expansion_iterations):
        power_high = analysis_of_covariance_power(
            effect_size_upper,
            sample_size_clamped,
            groups_clamped,
            covariate_r2_clamped,
            num_covariates_clamped,
            alpha,
        )
        needs_expansion = power_high < power
        if not torch.any(needs_expansion):
            break
        effect_size_upper = torch.where(
            needs_expansion, effect_size_upper * 2.0, effect_size_upper
        )
        effect_size_upper = torch.clamp(
            effect_size_upper, max=torch.tensor(10.0, dtype=dtype)
        )

    max_bisection_iterations = 24

    effect_size_mid = (effect_size_lower + effect_size_upper) * 0.5

    for _ in range(max_bisection_iterations):
        power_mid = analysis_of_covariance_power(
            effect_size_mid,
            sample_size_clamped,
            groups_clamped,
            covariate_r2_clamped,
            num_covariates_clamped,
            alpha,
        )
        power_too_low = power_mid < power

        effect_size_lower = torch.where(
            power_too_low, effect_size_mid, effect_size_lower
        )
        effect_size_upper = torch.where(
            power_too_low, effect_size_upper, effect_size_mid
        )

        effect_size_mid = (effect_size_lower + effect_size_upper) * 0.5

    result = torch.clamp(effect_size_mid, min=0.0)

    if scalar_output:
        result_scalar = result.reshape(())
        if out is not None:
            out.copy_(result_scalar)
            return out
        return result_scalar
    else:
        if out is not None:
            out.copy_(result)
            return out
        return result
