import math

import torch
from torch import Tensor


def z_test_sample_size(
    effect_size: Tensor,
    power: Tensor | float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    power = torch.atleast_1d(torch.as_tensor(power))

    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    dtype = torch.float32
    for tensor in (effect_size, power):
        dtype = torch.promote_types(dtype, tensor.dtype)

    effect_size = effect_size.to(dtype)

    power = power.to(dtype)

    power = torch.clamp(power, min=1e-6, max=1.0 - 1e-6)

    abs_effect_size = torch.clamp(torch.abs(effect_size), min=1e-6)

    square_root_two = math.sqrt(2.0)

    z_beta = torch.erfinv(power) * square_root_two

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two
    elif alternative in ["larger", "smaller"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}",
        )

    sample_size = ((z_alpha + z_beta) / abs_effect_size) ** 2

    result = torch.ceil(sample_size)

    result = torch.clamp(result, min=1.0)

    if out is not None:
        out.copy_(result)
        return out

