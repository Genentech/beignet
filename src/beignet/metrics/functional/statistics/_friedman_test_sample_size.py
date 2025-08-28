"""friedman test sample size functional metric."""

import beignet.statistics


def friedman_test_sample_size(*args, **kwargs):
    """
    Compute friedman_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.friedman_test_sample_size(*args, **kwargs)
