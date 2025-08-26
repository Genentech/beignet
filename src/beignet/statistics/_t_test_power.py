import torch
from torch import Tensor


def t_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if effect_size.dtype == torch.float64 or sample_size.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)

    degrees_of_freedom = sample_size - 1

    noncentrality = effect_size * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">", "one-sided", "one_sided"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_eff = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

        t_critical = z_eff * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
    else:
        z_eff = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

        t_critical = z_eff * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    mean_nct = noncentrality

    variance_nct = torch.where(
        degrees_of_freedom > 2,
        (degrees_of_freedom + noncentrality**2) / (degrees_of_freedom - 2),
        1 + noncentrality**2 / (2 * torch.clamp(degrees_of_freedom, min=2.0)),
    )
    standard_deviation_nct = torch.sqrt(variance_nct)

    if alt == "two-sided":
        z_upper = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct, min=1e-10
        )

        z_lower = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct, min=1e-10
        )

        power = 0.5 * (1 - torch.erf(z_upper / torch.sqrt(torch.tensor(2.0)))) + 0.5 * (
            1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0)))
        )
    elif alt == "greater":
        z_score = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct, min=1e-10
        )

        power = 0.5 * (1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))
    else:
        z_score = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct, min=1e-10
        )

        power = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
