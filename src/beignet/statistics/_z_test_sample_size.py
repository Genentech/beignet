import torch
from torch import Tensor
from torch.distributions import Normal


def z_test_sample_size(
    input: Tensor,
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

    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    dtype = torch.float32
    for tensor in (input, power):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    power = power.to(dtype)

    power = torch.clamp(power, min=1e-6, max=1.0 - 1e-6)

    abs_effect_size = torch.clamp(torch.abs(input), min=1e-6)

    normal_dist = Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    z_beta = normal_dist.icdf(power)

    if alternative == "two-sided":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    elif alternative in ["larger", "smaller"]:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}",
        )

    sample_size = ((z_alpha + z_beta) / abs_effect_size) ** 2

    result = torch.clamp(torch.ceil(sample_size), min=1.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
