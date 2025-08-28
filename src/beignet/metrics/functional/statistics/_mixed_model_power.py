"""mixed model power functional metric."""

import beignet.statistics


def mixed_model_power(*args, **kwargs):
    """
    Compute mixed_model_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.mixed_model_power(*args, **kwargs)
