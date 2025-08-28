"""chi squared goodness of fit sample size functional metric."""

import beignet.statistics


def chi_squared_goodness_of_fit_sample_size(*args, **kwargs):
    """
    Compute chi squared goodness of fit sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.chi_squared_goodness_of_fit_sample_size(*args, **kwargs)
