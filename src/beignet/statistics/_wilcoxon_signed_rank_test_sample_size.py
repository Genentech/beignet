import math

import torch
from torch import Tensor

import beignet.distributions

from ._wilcoxon_signed_rank_test_power import wilcoxon_signed_rank_test_power


def wilcoxon_signed_rank_test_sample_size(
    prob_positive: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    probability = torch.atleast_1d(torch.as_tensor(prob_positive))

    dtype = torch.float64 if probability.dtype == torch.float64 else torch.float32

    probability = probability.to(dtype)

    sqrt3 = math.sqrt(3.0)
    normal_dist = beignet.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    alt = alternative.lower()

    if alt in {"larger", "greater", ">"}:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))
    elif alt in {"smaller", "less", "<"}:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))
    elif alt == "two-sided":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    n_curr = torch.clamp(
        (
            (z_alpha + normal_dist.icdf(torch.tensor(power, dtype=dtype)))
            / (sqrt3 * torch.clamp(torch.abs(probability - 0.5), min=1e-8))
        )
        ** 2,
        min=5.0,
    )

    for _ in range(12):
        n_curr = torch.clamp(
            n_curr
            * (
                1.0
                + 1.25
                * torch.clamp(
                    power
                    - wilcoxon_signed_rank_test_power(
                        probability,
                        torch.ceil(n_curr),
                        alpha=alpha,
                        alternative=alt,
                    ),
                    min=-0.45,
                    max=0.45,
                )
            ),
            min=5.0,
            max=1e7,
        )

    n_out = torch.ceil(n_curr)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
