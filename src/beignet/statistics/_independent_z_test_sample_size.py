import math

import torch
from torch import Tensor


def independent_z_test_sample_size(
    input: Tensor,
    ratio: Tensor | None = None,
    power: Tensor | float = 0.8,
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
    ratio : Tensor | None, optional
        Sample size ratio.
    power : Tensor | float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    power = torch.atleast_1d(torch.as_tensor(power))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    if (
        input.dtype == torch.float64
        or ratio.dtype == torch.float64
        or power.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    ratio = ratio.to(dtype)
    power = power.to(dtype)

    power = torch.clamp(power, min=torch.finfo(dtype).eps, max=1.0 - 1e-6)

    abs_effect_size = torch.clamp(torch.abs(input), min=torch.finfo(dtype).eps)

    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    square_root_two = math.sqrt(2.0)

    z_beta = torch.erfinv(power) * square_root_two

    if alternative == "two-sided":
        z_alpha = (
            torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two
        )
    elif alternative in ["larger", "smaller"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}",
        )

    variance_scaling = 1 + 1 / ratio

    sample_size1 = ((z_alpha + z_beta) / abs_effect_size) ** 2 * variance_scaling

    result = torch.ceil(sample_size1)

    result = torch.clamp(result, min=1.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
