from typing import Callable
import torch


def _map_neighbor(metric_or_displacement: Callable
                 ) -> Callable:
    """Vectorizes a metric or displacement function over neighborhoods."""
    def wrapped_fn(Ra, Rb, **kwargs):
        return torch.vmap(torch.vmap(metric_or_displacement, (0, None)))(Rb, Ra, **kwargs)

    return wrapped_fn
