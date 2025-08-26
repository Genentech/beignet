import math

import torch
from torch import Tensor


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
    r"""
    """
    auc = torch.atleast_1d(torch.as_tensor(auc))

    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
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
        sample_size_group_2 = torch.atleast_1d(torch.as_tensor(nobs2))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64
            for t in (auc, sample_size_group_1, sample_size_group_2)
        )
        else torch.float32
    )
    auc = auc.to(dtype)

    sample_size_group_1 = torch.clamp(sample_size_group_1.to(dtype), min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2.to(dtype), min=2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    mean0 = sample_size_group_1 * sample_size_group_2 / 2.0

    var0 = (
        sample_size_group_1
        * sample_size_group_2
        * (sample_size_group_1 + sample_size_group_2 + 1.0)
        / 12.0
    )
    mean1 = sample_size_group_1 * sample_size_group_2 * auc

    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))

    noncentrality = (mean1 - mean0) / sd0

    square_root_two = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
    r"""
    """
        pt = torch.tensor(p, dtype=dtype)

        eps = torch.finfo(dtype).eps

        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return square_root_two * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        z_critical = z_of(1 - alpha / 2)

        upper = 0.5 * (1 - torch.erf((z_critical - noncentrality) / math.sqrt(2.0)))
        lower = 0.5 * (1 + torch.erf((-z_critical - noncentrality) / math.sqrt(2.0)))
        power = upper + lower
    elif alt == "greater":
        z_critical = z_of(1 - alpha)

        power = 0.5 * (1 - torch.erf((z_critical - noncentrality) / math.sqrt(2.0)))
    else:
        z_critical = z_of(1 - alpha)

        power = 0.5 * (1 + torch.erf((-z_critical - noncentrality) / math.sqrt(2.0)))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
