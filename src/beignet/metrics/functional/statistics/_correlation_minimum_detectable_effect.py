"""correlation minimum detectable effect functional metric."""

import beignet.statistics


def correlation_minimum_detectable_effect(*args, **kwargs):
    """
    Compute correlation_minimum_detectable_effect.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.correlation_minimum_detectable_effect(*args, **kwargs)
