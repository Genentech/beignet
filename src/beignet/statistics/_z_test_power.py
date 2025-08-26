import torch
from torch import Tensor


def z_test_power(
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
    alternative : str, default "two-sided"
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

    sample_size = torch.clamp(sample_size, min=1.0)

    noncentrality = input * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    square_root_two = torch.sqrt(torch.tensor(2.0, dtype=dtype))

    def z_of(p):
        pt = torch.as_tensor(p, dtype=dtype)

        eps = torch.finfo(dtype).eps

        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return square_root_two * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        z_alpha_half = z_of(1 - alpha / 2)

        power_upper = 0.5 * (
            1 - torch.erf((z_alpha_half - noncentrality) / square_root_two)
        )
        power_lower = 0.5 * (
            1 + torch.erf((-z_alpha_half - noncentrality) / square_root_two)
        )
        power = power_upper + power_lower
    elif alt == "greater":
        z_alpha = z_of(1 - alpha)

        power = 0.5 * (1 - torch.erf((z_alpha - noncentrality) / square_root_two))
    else:
        z_alpha = z_of(1 - alpha)

        power = 0.5 * (1 + torch.erf((-z_alpha - noncentrality) / square_root_two))

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
