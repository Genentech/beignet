import torch
from torch import Tensor

import beignet.distributions

from ._wilcoxon_signed_rank_test_power import wilcoxon_signed_rank_test_power


def wilcoxon_signed_rank_test_minimum_detectable_effect(
    nobs: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    sample_size_0 = torch.as_tensor(nobs)

    sample_size = torch.atleast_1d(sample_size_0)

    dtype = torch.float64 if sample_size.dtype == torch.float64 else torch.float32

    sample_size = torch.clamp(sample_size.to(dtype), min=5.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    s = sample_size * (sample_size + 1.0) / 2.0

    var0 = sample_size * (sample_size + 1.0) * (2.0 * sample_size + 1.0) / 24.0

    sd0 = torch.sqrt(torch.maximum(var0, torch.finfo(dtype).eps))

    normal_dist = beignet.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    z_alpha = (
        normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
        if alt == "two-sided"
        else normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))
    )

    z_beta = normal_dist.icdf(torch.tensor(power, dtype=dtype))

    delta = (z_alpha + z_beta) * sd0 / torch.maximum(s, torch.finfo(dtype).eps)

    if alt == "less":
        prob_initial = torch.clamp(0.5 - delta, 0.0, 1.0)
    else:
        prob_initial = torch.clamp(0.5 + delta, 0.0, 1.0)

    if alt == "less":
        prob_lo = torch.zeros_like(prob_initial)

        prob_hi = torch.full_like(prob_initial, 0.5)
    else:
        prob_lo = torch.full_like(prob_initial, 0.5)

        prob_hi = torch.ones_like(prob_initial)

    if alt == "less":
        max_power_prob = prob_lo
    else:
        max_power_prob = prob_hi

    max_power = wilcoxon_signed_rank_test_power(
        max_power_prob,
        sample_size,
        alpha=alpha,
        alternative=alt,
    )

    unattainable = max_power < power - 1e-6

    probability = (prob_lo + prob_hi) * 0.5
    for _ in range(24):
        current_power = wilcoxon_signed_rank_test_power(
            probability,
            sample_size,
            alpha=alpha,
            alternative=alt,
        )
        too_low = current_power < power

        if alt == "less":
            prob_hi = torch.where(too_low, probability, prob_hi)
            prob_lo = torch.where(too_low, prob_lo, probability)
        else:
            prob_lo = torch.where(too_low, probability, prob_lo)
            prob_hi = torch.where(too_low, prob_hi, probability)

        probability = (prob_lo + prob_hi) * 0.5

    probability = torch.where(unattainable, max_power_prob, probability)

    if out is not None:
        out.copy_(probability)

        return out

    return probability
