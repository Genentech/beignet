import torch
from torch import Tensor


def t_test_power(
    input: Tensor,
    sample_size: Tensor,
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

    input = torch.atleast_1d(torch.as_tensor(input))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    dtype = torch.float32
    for tensor in (input, sample_size):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)

    degrees_of_freedom = sample_size - 1

    noncentrality = input * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">", "one-sided", "one_sided"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    square_root_two = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_eff = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two

        t_critical = z_eff * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
    else:
        z_eff = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

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
            standard_deviation_nct,
            min=1e-10,
        )

        z_lower = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (1 - torch.erf(z_upper / square_root_two)) + 0.5 * (
            1 + torch.erf(z_lower / square_root_two)
        )
    elif alt == "greater":
        z_score = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (1 - torch.erf(z_score / square_root_two))
    else:
        z_score = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (1 + torch.erf(z_score / square_root_two))

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out

    return result
