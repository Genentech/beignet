import math

import torch
from torch import Tensor


def independent_t_test_sample_size(
    effect_size: Tensor,
    ratio: Tensor | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    dtype = torch.float32
    for tensor in (effect_size, ratio):
        dtype = torch.promote_types(dtype, tensor.dtype)

    effect_size = effect_size.to(dtype)

    ratio = ratio.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    variance_scaling = (1 + 1 / ratio) / 2

    sample_size_group_1_initial = (
        (z_alpha + z_beta) / effect_size
    ) ** 2 * variance_scaling

    sample_size_group_1_initial = torch.clamp(sample_size_group_1_initial, min=2.0)

    sample_size_group_1_iteration = sample_size_group_1_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        sample_size_group_2_iteration = sample_size_group_1_iteration * ratio

        total_n = sample_size_group_1_iteration + sample_size_group_2_iteration

        degrees_of_freedom_iteration = total_n - 2

        degrees_of_freedom_iteration = torch.clamp(
            degrees_of_freedom_iteration,
            min=1.0,
        )

        se_factor = torch.sqrt(
            1 / sample_size_group_1_iteration + 1 / sample_size_group_2_iteration,
        )

        ncp_iteration = effect_size / se_factor

        if alternative == "two-sided":
            t_critical = z_alpha * torch.sqrt(
                1 + 1 / (2 * degrees_of_freedom_iteration),
            )
            t_critical = z_alpha * torch.sqrt(
                1 + 1 / (2 * degrees_of_freedom_iteration),
            )

        variance_nct = torch.where(
            degrees_of_freedom_iteration > 2,
            (degrees_of_freedom_iteration + ncp_iteration**2)
            / (degrees_of_freedom_iteration - 2),
            1 + ncp_iteration**2 / (2 * degrees_of_freedom_iteration),
        )
        standard_deviation_nct = torch.sqrt(variance_nct)

        if alternative == "two-sided":
            z_upper = (t_critical - ncp_iteration) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            z_lower = (-t_critical - ncp_iteration) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            power_iteration = (1 - torch.erf(z_upper / sqrt_2)) / 2 + (
                1 - torch.erf(-z_lower / sqrt_2)
            ) / 2
        elif alternative == "larger":
            z_score = (t_critical - ncp_iteration) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            power_iteration = (1 - torch.erf(z_score / sqrt_2)) / 2
            z_score = (-t_critical - (-ncp_iteration)) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )

            power_iteration = (1 - torch.erf(-z_score / sqrt_2)) / 2

        power_iteration = torch.clamp(power_iteration, 0.01, 0.99)

        power_diff = power - power_iteration

        adjustment = (
            power_diff
            * sample_size_group_1_iteration
            / (2 * torch.clamp(power_iteration * (1 - power_iteration), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        sample_size_group_1_iteration = sample_size_group_1_iteration + adjustment

        sample_size_group_1_iteration = torch.clamp(
            sample_size_group_1_iteration,
            min=2.0,
            max=100000.0,
        )

    result = torch.ceil(sample_size_group_1_iteration)

    result = torch.clamp(result, min=2.0)

    if out is not None:
        out.copy_(result)
        return out

