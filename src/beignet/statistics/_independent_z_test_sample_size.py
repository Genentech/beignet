import math

import torch
from torch import Tensor


def independent_z_test_sample_size(
    effect_size: Tensor,
    ratio: Tensor | None = None,
    power: Tensor | float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    power = torch.atleast_1d(torch.as_tensor(power))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    if (
        effect_size.dtype == torch.float64
        or ratio.dtype == torch.float64
        or power.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    ratio = ratio.to(dtype)
    power = power.to(dtype)

    power = torch.clamp(power, min=1e-6, max=1.0 - 1e-6)

    abs_effect_size = torch.clamp(torch.abs(effect_size), min=1e-6)

    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sqrt_2 = math.sqrt(2.0)

    z_beta = torch.erfinv(power) * sqrt_2

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
    elif alternative in ["larger", "smaller"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}"
        )

    variance_factor = 1 + 1 / ratio

    sample_size1 = ((z_alpha + z_beta) / abs_effect_size) ** 2 * variance_factor

    output = torch.ceil(sample_size1)

    output = torch.clamp(output, min=1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
