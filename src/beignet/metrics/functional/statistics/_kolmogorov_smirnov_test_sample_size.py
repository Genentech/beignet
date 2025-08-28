"""kolmogorov smirnov test sample size functional metric."""

import beignet.statistics


def kolmogorov_smirnov_test_sample_size(*args, **kwargs):
    """
    Compute kolmogorov_smirnov_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.kolmogorov_smirnov_test_sample_size(*args, **kwargs)
