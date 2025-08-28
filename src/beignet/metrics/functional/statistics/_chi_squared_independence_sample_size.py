"""chi squared independence sample size functional metric."""

import beignet.statistics


def chi_squared_independence_sample_size(*args, **kwargs):
    """
    Compute chi squared independence sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.chi_square_independence_sample_size(*args, **kwargs)
