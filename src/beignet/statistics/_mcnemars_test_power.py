import torch
from torch import Tensor

import beignet.distributions


def mcnemars_test_power(
    p01: Tensor,
    p10: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    p01 = torch.atleast_1d(p01)
    p10 = torch.atleast_1d(p10)

    sample_size = torch.atleast_1d(sample_size)

    dtype = torch.promote_types(p01.dtype, p10.dtype)
    dtype = torch.promote_types(dtype, sample_size.dtype)
    p01 = torch.clamp(p01.to(dtype), 0.0, 1.0)
    p10 = torch.clamp(p10.to(dtype), 0.0, 1.0)

    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    if two_sided:
        power = (
            1
            - beignet.distributions.StandardNormal.from_dtype(dtype).cdf(
                beignet.distributions.StandardNormal.from_dtype(dtype).icdf(
                    torch.tensor(1 - alpha / 2, dtype=dtype),
                )
                - sample_size
                * (p01 + p10)
                * (
                    torch.where(
                        (p01 + p10) > 0,
                        p01 / torch.clamp(p01 + p10, min=torch.finfo(dtype).eps),
                        torch.zeros_like(p01),
                    )
                    - 0.5
                )
                / torch.clamp(
                    torch.sqrt(
                        torch.clamp(
                            sample_size * (p01 + p10) * 0.25,
                            min=torch.finfo(dtype).eps,
                        ),
                    ),
                    min=torch.finfo(dtype).eps,
                ),
            )
        ) + beignet.distributions.StandardNormal.from_dtype(dtype).cdf(
            -beignet.distributions.StandardNormal.from_dtype(dtype).icdf(
                torch.tensor(1 - alpha / 2, dtype=dtype),
            )
            - sample_size
            * (p01 + p10)
            * (
                torch.where(
                    (p01 + p10) > 0,
                    p01 / torch.clamp(p01 + p10, min=torch.finfo(dtype).eps),
                    torch.zeros_like(p01),
                )
                - 0.5
            )
            / torch.clamp(
                torch.sqrt(
                    torch.clamp(
                        sample_size * (p01 + p10) * 0.25,
                        min=torch.finfo(dtype).eps,
                    ),
                ),
                min=torch.finfo(dtype).eps,
            ),
        )
    else:
        power = 1 - beignet.distributions.StandardNormal.from_dtype(dtype).cdf(
            beignet.distributions.StandardNormal.from_dtype(dtype).icdf(
                torch.tensor(1 - alpha, dtype=dtype),
            )
            - sample_size
            * (p01 + p10)
            * (
                torch.where(
                    (p01 + p10) > 0,
                    p01 / torch.clamp(p01 + p10, min=torch.finfo(dtype).eps),
                    torch.zeros_like(p01),
                )
                - 0.5
            )
            / torch.clamp(
                torch.sqrt(
                    torch.clamp(
                        sample_size * (p01 + p10) * 0.25,
                        min=torch.finfo(dtype).eps,
                    ),
                ),
                min=torch.finfo(dtype).eps,
            ),
        )

    out_t = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(out_t)

        return out

    return out_t
