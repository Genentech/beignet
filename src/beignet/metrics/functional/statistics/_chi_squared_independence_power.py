"""chi squared independence power functional metric."""

import beignet.statistics


def chi_squared_independence_power(*args, **kwargs):
    """
    Compute chi squared independence power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.chi_squared_independence_power(*args, **kwargs)
