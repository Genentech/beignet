from typing import Sequence

import torch
from torch import Tensor


def global_distance_test(
    input: Tensor,
    other: Tensor,
    mask: Tensor,
    cutoffs: Sequence[float],
) -> Tensor:
    n = torch.sum(mask, dim=-1)

    y = input - other
    y = y**2
    y = torch.sum(y, dim=-1)
    y = torch.sqrt(y)

    scores = torch.zeros(len(cutoffs))

    for index, cutoff in enumerate(cutoffs):
        scores[index] = torch.mean(torch.sum((y <= cutoff) * mask, dim=-1) / n)

    return sum(scores) / len(scores)


def global_distance_test_ts(p1, p2, mask):
    return global_distance_test(p1, p2, mask, [1.0, 2.0, 4.0, 8.0])


def global_distance_test_ha(p1, p2, mask):
    return global_distance_test(p1, p2, mask, [0.5, 1.0, 2.0, 4.0])
