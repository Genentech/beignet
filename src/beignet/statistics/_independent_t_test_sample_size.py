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

    if effect_size.dtype == torch.float64 or ratio.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    ratio = ratio.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    variance_factor = (1 + 1 / ratio) / 2

    sample_size_group_1_initial = (
        (z_alpha + z_beta) / effect_size
    ) ** 2 * variance_factor

    sample_size_group_1_initial = torch.clamp(sample_size_group_1_initial, min=2.0)

    sample_size_group_1_current = sample_size_group_1_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        sample_size_group_2_current = sample_size_group_1_current * ratio

        total_n = sample_size_group_1_current + sample_size_group_2_current

        df_current = total_n - 2

        df_current = torch.clamp(df_current, min=1.0)

        se_factor = torch.sqrt(
            1 / sample_size_group_1_current + 1 / sample_size_group_2_current
        )

        ncp_current = effect_size / se_factor

        if alternative == "two-sided":
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))
        else:
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))

        var_nct = torch.where(
            df_current > 2,
            (df_current + ncp_current**2) / (df_current - 2),
            1 + ncp_current**2 / (2 * df_current),
        )
        std_nct = torch.sqrt(var_nct)

        if alternative == "two-sided":
            z_upper = (t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)

            z_lower = (-t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)

            power_current = (1 - torch.erf(z_upper / sqrt_2)) / 2 + (
                1 - torch.erf(-z_lower / sqrt_2)
            ) / 2
        elif alternative == "larger":
            z_score = (t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)

            power_current = (1 - torch.erf(z_score / sqrt_2)) / 2
        else:
            z_score = (-t_critical - (-ncp_current)) / torch.clamp(std_nct, min=1e-10)

            power_current = (1 - torch.erf(-z_score / sqrt_2)) / 2

        power_current = torch.clamp(power_current, 0.01, 0.99)

        power_diff = power - power_current

        adjustment = (
            power_diff
            * sample_size_group_1_current
            / (2 * torch.clamp(power_current * (1 - power_current), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        sample_size_group_1_current = sample_size_group_1_current + adjustment

        sample_size_group_1_current = torch.clamp(
            sample_size_group_1_current, min=2.0, max=100000.0
        )

    output = torch.ceil(sample_size_group_1_current)

    output = torch.clamp(output, min=2.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
