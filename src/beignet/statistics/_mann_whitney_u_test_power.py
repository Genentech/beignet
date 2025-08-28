import torch
from torch import Tensor

import beignet.distributions


def mann_whitney_u_test_power(
    auc: Tensor,
    nobs1: Tensor,
    nobs2: Tensor | None = None,
    ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    auc = torch.atleast_1d(auc)

    sample_size_group_1 = torch.atleast_1d(nobs1)
    if nobs2 is None:
        r = torch.as_tensor(ratio)

        sample_size_group_2 = torch.ceil(
            sample_size_group_1
            * (
                r.to(sample_size_group_1.dtype)
                if isinstance(r, Tensor)
                else torch.tensor(float(r), dtype=sample_size_group_1.dtype)
            ),
        )
    else:
        sample_size_group_2 = torch.atleast_1d(nobs2)

    dtype = torch.promote_types(auc.dtype, sample_size_group_1.dtype)
    dtype = torch.promote_types(dtype, sample_size_group_2.dtype)
    auc = auc.to(dtype)

    sample_size_group_1 = torch.clamp(sample_size_group_1.to(dtype), min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2.to(dtype), min=2.0)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    mean0 = sample_size_group_1 * sample_size_group_2 / 2.0

    var0 = (
        sample_size_group_1
        * sample_size_group_2
        * (sample_size_group_1 + sample_size_group_2 + 1.0)
        / 12.0
    )
    mean1 = sample_size_group_1 * sample_size_group_2 * auc

    sd0 = torch.sqrt(
        torch.maximum(var0, torch.tensor(torch.finfo(dtype).eps, dtype=dtype)),
    )

    noncentrality = (mean1 - mean0) / sd0
    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        upper = 1 - normal_dist.cdf(z_critical - noncentrality)
        lower = normal_dist.cdf(-z_critical - noncentrality)
        power = upper + lower
    elif alternative == "greater":
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = 1 - normal_dist.cdf(z_critical - noncentrality)
    else:
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = normal_dist.cdf(-z_critical - noncentrality)

    out_t = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(out_t)

        return out

    return out_t
