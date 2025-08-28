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

    power = torch.clamp(power, min=torch.finfo(dtype).eps, max=1.0 - 1e-6)

    distribution = Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    match alternative:
        case "two-sided":
            output = torch.tensor(1 - alpha / 2, dtype=dtype)
        case "greater" | "less":
            output = torch.tensor(1 - alpha / 1, dtype=dtype)
        case _:
            raise ValueError

    output = distribution.icdf(output)
    output = output + distribution.icdf(power)
    output = output / torch.clamp(torch.abs(input), min=torch.finfo(dtype).eps)
    output = output**2
    output = torch.ceil(output)
    output = torch.clamp(output, min=1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
