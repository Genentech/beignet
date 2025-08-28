"""independent t test sample size functional metric."""

import beignet.statistics


def independent_t_test_sample_size(*args, **kwargs):
    """
    Compute independent t test sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.independent_t_test_sample_size(*args, **kwargs)
