"""logistic regression sample size functional metric."""

import beignet.statistics


def logistic_regression_sample_size(*args, **kwargs):
    """
    Compute logistic_regression_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.logistic_regression_sample_size(*args, **kwargs)
