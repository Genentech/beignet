import torch
from torch import Tensor

from ._t_test_power import t_test_power


def t_test_sample_size(
    input: Tensor,
    power: float = 0.8,
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
    power : float, default 0.8
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

    if input.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    input = torch.maximum(input, torch.finfo(dtype).eps)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt not in {"two-sided", "one-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    square_root_two = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    if alt == "two-sided":
        z_alpha = (
            torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * square_root_two
        )
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    sample_size_iteration = torch.clamp(
        (
            (
                z_alpha
                + torch.sqrt(torch.tensor(2.0, dtype=dtype))
                * torch.erfinv(
                    2.0 * torch.as_tensor(power, dtype=dtype) - 1.0,
                )
            )
            / input
        )
        ** 2,
        min=2.0,
    )

    for _iteration in range(10):
        power_iteration = torch.clamp(
            t_test_power(
                input,
                sample_size_iteration,
                alpha,
                alternative,
            ),
            0.01,
            0.99,
        )

        sample_size_iteration = torch.clamp(
            sample_size_iteration
            + torch.where(
                torch.abs(power - power_iteration) < 1e-6,
                (
                    (power - power_iteration)
                    * sample_size_iteration
                    / (
                        2
                        * torch.clamp(power_iteration * (1 - power_iteration), min=0.01)
                    )
                )
                * 0.1,
                (
                    (power - power_iteration)
                    * sample_size_iteration
                    / (
                        2
                        * torch.clamp(power_iteration * (1 - power_iteration), min=0.01)
                    )
                ),
            ),
            min=2.0,
            max=100000.0,
        )

    result = torch.clamp(torch.ceil(sample_size_iteration), min=2.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
