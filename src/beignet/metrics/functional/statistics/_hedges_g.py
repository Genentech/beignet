"""Hedges' g effect size functional metric."""

import beignet.statistics


def hedges_g(*args, **kwargs):
    """
    Compute Hedges' g effect size between two groups.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.hedges_g(*args, **kwargs)
