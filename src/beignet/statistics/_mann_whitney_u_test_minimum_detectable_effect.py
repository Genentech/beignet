import math

import torch
from torch import Tensor

from ._mann_whitney_u_test_power import mann_whitney_u_test_power


def mann_whitney_u_test_minimum_detectable_effect(
    nobs1: Tensor,
    nobs2: Tensor | None = None,
    ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    sample_size_group_1_0 = torch.as_tensor(nobs1)

    scalar_out = sample_size_group_1_0.ndim == 0

    sample_size_group_1 = torch.atleast_1d(sample_size_group_1_0)
    if nobs2 is None:
        r = torch.as_tensor(ratio)

        sample_size_group_2 = torch.ceil(
            sample_size_group_1
            * (
                r.to(sample_size_group_1.dtype)
                if isinstance(r, Tensor)
                else torch.tensor(float(r), dtype=sample_size_group_1.dtype)
            )
        )
    else:
        sample_size_group_2_0 = torch.as_tensor(nobs2)

        scalar_out = scalar_out and (sample_size_group_2_0.ndim == 0)
        sample_size_group_2 = torch.atleast_1d(sample_size_group_2_0)

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64 for t in (sample_size_group_1, sample_size_group_2)
        )
        else torch.float32
    )
    sample_size_group_1 = torch.clamp(sample_size_group_1.to(dtype), min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2.to(dtype), min=2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
        q = torch.tensor(p, dtype=dtype)

        eps = torch.finfo(dtype).eps

        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    z_alpha = z_of(1 - alpha / 2) if alt == "two-sided" else z_of(1 - alpha)

    z_beta = z_of(power)

    scale = torch.sqrt(
        12.0
        * sample_size_group_1
        * sample_size_group_2
        / (sample_size_group_1 + sample_size_group_2 + 1.0)
    )
    delta = (z_alpha + z_beta) / torch.clamp(scale, min=1e-12)

    if alt == "less":
        auc = torch.clamp(0.5 - delta, min=0.0)
    else:
        auc = torch.clamp(0.5 + delta, max=1.0)

    p_curr = mann_whitney_u_test_power(
        auc, sample_size_group_1, sample_size_group_2, alpha=alpha, alternative=alt
    )
    gap = torch.clamp(power - p_curr, min=-0.45, max=0.45)

    step = gap * 0.05
    if alt == "less":
        auc = torch.clamp(auc - torch.abs(step), 0.0, 1.0)
    else:
        auc = torch.clamp(auc + torch.abs(step), 0.0, 1.0)

    if scalar_out:
        auc_s = auc.reshape(())
        if out is not None:
            out.copy_(auc_s)
            return out
        return auc_s
    else:
        if out is not None:
            out.copy_(auc)
            return out
        return auc
