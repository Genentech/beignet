"""multivariate analysis of variance power functional metric."""

import beignet.statistics


def multivariate_analysis_of_variance_power(*args, **kwargs):
    """
    Compute multivariate_analysis_of_variance_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.multivariate_analysis_of_variance_power(*args, **kwargs)
