from typing import Callable

import torch


def _map_bond(metric_or_displacement: Callable) -> Callable:
    r"""Map a distance function over batched start and end positions.

    Parameters:
    -----------
    distance_fn : callable
        A function that computes the distance between two positions.

    Returns:
    --------
    wrapper : callable
        A wrapper function that applies `distance_fn` to each pair of start and end positions
        in the batch.
    """
    return torch.vmap(metric_or_displacement, (0, 0), 0)
