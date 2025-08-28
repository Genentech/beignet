"""proportion power functional metric."""

import beignet.statistics


def proportion_power(*args, **kwargs):
    """
    Compute proportion power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.proportion_power(*args, **kwargs)
