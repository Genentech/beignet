"""mcnemars test sample size functional metric."""

import beignet.statistics


def mcnemars_test_sample_size(*args, **kwargs):
    """
    Compute mcnemars_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.mcnemars_test_sample_size(*args, **kwargs)
