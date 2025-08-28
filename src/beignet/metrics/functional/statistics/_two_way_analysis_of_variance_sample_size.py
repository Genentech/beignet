"""two way analysis of variance sample size functional metric."""

import beignet.statistics


def two_way_analysis_of_variance_sample_size(*args, **kwargs):
    """
    Compute two_way_analysis_of_variance_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.two_way_analysis_of_variance_sample_size(*args, **kwargs)
