"""correlation sample size functional metric."""

import beignet.statistics


def correlation_sample_size(*args, **kwargs):
    """
    Compute correlation sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.correlation_sample_size(*args, **kwargs)
