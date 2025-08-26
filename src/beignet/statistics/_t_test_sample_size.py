import torch
from torch import Tensor


def t_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    if effect_size.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    square_root_two = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.sqrt(torch.tensor(2.0, dtype=dtype)) * torch.erfinv(
        2.0 * torch.as_tensor(power, dtype=dtype) - 1.0,
    )

    sample_size_initial = ((z_alpha + z_beta) / effect_size) ** 2

    sample_size_initial = torch.clamp(sample_size_initial, min=2.0)

    sample_size_iteration = sample_size_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        degrees_of_freedom_iteration = sample_size_iteration - 1

        degrees_of_freedom_iteration = torch.clamp(
            degrees_of_freedom_iteration,
            min=1.0,
        )

        noncentrality_iteration = effect_size * torch.sqrt(sample_size_iteration)

        if alternative == "two-sided":
            t_critical = z_alpha * torch.sqrt(
                1 + 1 / (2 * degrees_of_freedom_iteration),
            )
        else:
            t_critical = z_alpha * torch.sqrt(
                1 + 1 / (2 * degrees_of_freedom_iteration),
            )

        variance_nct = (
            degrees_of_freedom_iteration + noncentrality_iteration**2
        ) / torch.clamp(degrees_of_freedom_iteration - 2, min=0.1)
        variance_nct = torch.where(
            degrees_of_freedom_iteration > 2,
            variance_nct,
            1 + noncentrality_iteration**2 / (2 * degrees_of_freedom_iteration),
        )
        standard_deviation_nct = torch.sqrt(variance_nct)

        if alternative == "two-sided":
            z_upper = (t_critical - noncentrality_iteration) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )
            z_lower = (-t_critical - noncentrality_iteration) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )
            power_iteration = 0.5 * (
                1 - torch.erf(z_upper / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (
                1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            z_score = (t_critical - noncentrality_iteration) / torch.clamp(
                standard_deviation_nct,
                min=1e-10,
            )
            power_iteration = 0.5 * (
                1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )

        power_iteration = torch.clamp(power_iteration, 0.01, 0.99)

        power_diff = power - power_iteration

        adjustment = (
            power_diff
            * sample_size_iteration
            / (2 * torch.clamp(power_iteration * (1 - power_iteration), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)

        sample_size_iteration = sample_size_iteration + adjustment

        sample_size_iteration = torch.clamp(
            sample_size_iteration,
            min=2.0,
            max=100000.0,
        )

    result = torch.ceil(sample_size_iteration)

    result = torch.clamp(result, min=2.0)

    if out is not None:
        out.copy_(result)
        return out

