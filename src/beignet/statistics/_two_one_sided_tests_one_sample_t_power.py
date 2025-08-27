import torch
from torch import Tensor

import beignet.distributions


def two_one_sided_tests_one_sample_t_power(
    true_effect: Tensor,
    sample_size: Tensor,
    low: Tensor,
    high: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    true_effect_size = torch.atleast_1d(torch.as_tensor(true_effect))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    low = torch.atleast_1d(torch.as_tensor(low))

    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64 for t in (true_effect_size, sample_size, low, high)
        )
        else torch.float32
    )
    true_effect_size = true_effect_size.to(dtype)

    sample_size = sample_size.to(dtype)

    low = low.to(dtype)

    high = high.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)

    degrees_of_freedom = sample_size - 1

    t_critical = beignet.distributions.StudentT(degrees_of_freedom).icdf(
        torch.tensor(1 - alpha, dtype=dtype),
    )

    def power_greater(noncentrality: Tensor) -> Tensor:
        return 0.5 * (
            1
            - torch.erf(
                (
                    t_critical
                    - beignet.distributions.NonCentralT(
                        degrees_of_freedom,
                        noncentrality,
                    ).mean
                )
                / torch.clamp(
                    torch.sqrt(
                        beignet.distributions.NonCentralT(
                            degrees_of_freedom,
                            noncentrality,
                        ).variance,
                    ),
                    min=1e-10,
                )
                / torch.sqrt(torch.tensor(2.0, dtype=dtype)),
            )
        )

    def power_less(noncentrality: Tensor) -> Tensor:
        return 0.5 * (
            1
            + torch.erf(
                (
                    -t_critical
                    - beignet.distributions.NonCentralT(
                        degrees_of_freedom,
                        noncentrality,
                    ).mean
                )
                / torch.clamp(
                    torch.sqrt(
                        beignet.distributions.NonCentralT(
                            degrees_of_freedom,
                            noncentrality,
                        ).variance,
                    ),
                    min=1e-10,
                )
                / torch.sqrt(torch.tensor(2.0, dtype=dtype)),
            )
        )

    power = torch.clamp(
        torch.minimum(
            power_greater((true_effect_size - low) * torch.sqrt(sample_size)),
            power_less((true_effect_size - high) * torch.sqrt(sample_size)),
        ),
        0.0,
        1.0,
    )

    if out is not None:
        out.copy_(power)

        return out

    return power
