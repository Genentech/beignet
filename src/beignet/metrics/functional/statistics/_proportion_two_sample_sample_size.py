"""proportion two sample sample size functional metric."""

import beignet.statistics


def proportion_two_sample_sample_size(*args, **kwargs):
    """
    Compute proportion two sample sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.proportion_two_sample_sample_size(*args, **kwargs)
