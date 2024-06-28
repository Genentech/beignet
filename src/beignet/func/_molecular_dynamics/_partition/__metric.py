from typing import Callable

import torch

from src.beignet.func._molecular_dynamics._partition.__distance import distance


def metric(distance_fn: Callable) -> Callable:
    r"""Takes a displacement function and creates a metric..

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
    return lambda Ra, Rb, **kwargs: distance(distance_fn(Ra, Rb, **kwargs))
