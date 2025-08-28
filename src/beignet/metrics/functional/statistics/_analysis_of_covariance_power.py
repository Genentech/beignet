"""analysis of covariance power functional metric."""

import beignet.statistics


def analysis_of_covariance_power(*args, **kwargs):
    """
    Compute analysis_of_covariance_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.analysis_of_covariance_power(*args, **kwargs)
