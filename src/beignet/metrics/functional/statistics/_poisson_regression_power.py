"""poisson regression power functional metric."""

import beignet.statistics


def poisson_regression_power(*args, **kwargs):
    """
    Compute poisson_regression_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.poisson_regression_power(*args, **kwargs)
