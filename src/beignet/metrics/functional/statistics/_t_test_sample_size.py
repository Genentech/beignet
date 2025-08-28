"""t test sample size functional metric."""

import beignet.statistics


def t_test_sample_size(*args, **kwargs):
    """
    Compute t test sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.t_test_sample_size(*args, **kwargs)
