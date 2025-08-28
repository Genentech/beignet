"""repeated measures analysis of variance sample size functional metric."""

import beignet.statistics


def repeated_measures_analysis_of_variance_sample_size(*args, **kwargs):
    """
    Compute repeated_measures_analysis_of_variance_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.repeated_measures_analysis_of_variance_sample_size(
        *args, **kwargs
    )
