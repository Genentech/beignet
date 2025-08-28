"""analysis of covariance sample size functional metric."""

import beignet.statistics


def analysis_of_covariance_sample_size(*args, **kwargs):
    """
    Compute analysis_of_covariance_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.analysis_of_covariance_sample_size(*args, **kwargs)
