"""chi squared goodness of fit power functional metric."""

import beignet.statistics


def chi_squared_goodness_of_fit_power(*args, **kwargs):
    """
    Compute chi squared goodness of fit power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.chi_square_goodness_of_fit_power(*args, **kwargs)
