"""mixed model sample size functional metric."""

import beignet.statistics


def mixed_model_sample_size(*args, **kwargs):
    """
    Compute mixed_model_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.mixed_model_sample_size(*args, **kwargs)
