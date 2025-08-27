import torch
from torch import Tensor

import beignet.distributions

from ._mann_whitney_u_test_power import mann_whitney_u_test_power


def mann_whitney_u_test_sample_size(
    auc: Tensor,
    ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    auc = torch.atleast_1d(torch.as_tensor(auc))

    r = torch.as_tensor(ratio)

    dtype = (
        torch.float64
        if (auc.dtype == torch.float64 or r.dtype == torch.float64)
        else torch.float32
    )
    auc = auc.to(dtype)

    r = torch.clamp(r.to(dtype), min=0.1, max=10.0)

    normal_dist = beignet.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    z_alpha = normal_dist.icdf(
        torch.tensor(
            1 - (alpha / 2 if alternative == "two-sided" else alpha),
            dtype=dtype,
        ),
    )

    z_beta = normal_dist.icdf(torch.tensor(power, dtype=dtype))

    delta = torch.abs(auc - 0.5)

    c = torch.sqrt(1.0 + r) / torch.sqrt(12.0 * r)

    sample_size_group_1 = ((z_alpha + z_beta) * c / torch.clamp(delta, min=1e-8)) ** 2

    sample_size_group_1 = torch.clamp(sample_size_group_1, min=5.0)

    sample_size_group_1_iteration = sample_size_group_1
    for _ in range(12):
        pwr = mann_whitney_u_test_power(
            auc,
            torch.ceil(sample_size_group_1_iteration),
            ratio=r,
            alpha=alpha,
            alternative=alternative,
        )
        gap = torch.clamp(power - pwr, min=-0.45, max=0.45)

        sample_size_group_1_iteration = torch.clamp(
            sample_size_group_1_iteration * (1.0 + 1.25 * gap),
            min=5.0,
            max=1e7,
        )

    sample_size_group_1_result = torch.ceil(sample_size_group_1_iteration)

    if out is not None:
        out.copy_(sample_size_group_1_result)

        return out

    return sample_size_group_1_result
