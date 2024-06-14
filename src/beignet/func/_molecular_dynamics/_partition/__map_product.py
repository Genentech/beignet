import torch

from typing import Callable


def map_product(metric_or_displacement: Callable) -> Callable:
    """Vectorizes a metric or displacement function over all pairs."""
    outer_vmap = torch.vmap(torch.vmap(metric_or_displacement, (0, None)), (None, 0))

    return outer_vmap
