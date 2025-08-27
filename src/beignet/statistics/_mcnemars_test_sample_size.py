import torch
from torch import Tensor

import beignet.distributions

from ._mcnemars_test_power import mcnemars_test_power


def mcnemars_test_sample_size(
    p01: Tensor,
    p10: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))

    dtype = (
        torch.float64
        if (p01.dtype == torch.float64 or p10.dtype == torch.float64)
        else torch.float32
    )
    p01 = p01.to(dtype)
    p10 = p10.to(dtype)

    normal_dist = beignet.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    if two_sided:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    n_curr = torch.clamp(
        (
            (z_alpha + normal_dist.icdf(torch.tensor(power, dtype=dtype)))
            / (
                2
                * torch.clamp(
                    torch.abs(
                        torch.where(
                            (p01 + p10) > 0,
                            p01 / torch.clamp(p01 + p10, min=1e-12),
                            torch.zeros_like(p01),
                        )
                        - 0.5,
                    ),
                    min=1e-8,
                )
            )
        )
        ** 2
        / torch.clamp(
            p01 + p10,
            min=1e-8,
        ),
        min=4.0,
    )

    for _ in range(12):
        n_curr = torch.clamp(
            n_curr
            * (
                1.0
                + 1.25
                * torch.clamp(
                    power
                    - mcnemars_test_power(
                        p01,
                        p10,
                        torch.ceil(n_curr),
                        alpha=alpha,
                        two_sided=two_sided,
                    ),
                    min=-0.45,
                    max=0.45,
                )
            ),
            min=4.0,
            max=1e7,
        )

    n_out = torch.ceil(n_curr)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
