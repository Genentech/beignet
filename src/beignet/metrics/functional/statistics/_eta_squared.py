"""eta squared functional metric."""

import beignet.statistics


def eta_squared(*args, **kwargs):
    """
    Compute eta_squared.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.eta_squared(*args, **kwargs)
