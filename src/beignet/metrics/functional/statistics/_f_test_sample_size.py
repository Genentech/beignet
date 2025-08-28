"""f test sample size functional metric."""

import beignet.statistics


def f_test_sample_size(*args, **kwargs):
    """
    Compute f test sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.f_test_sample_size(*args, **kwargs)
