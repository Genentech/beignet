"""logistic regression power functional metric."""

import beignet.statistics


def logistic_regression_power(*args, **kwargs):
    """
    Compute logistic_regression_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.logistic_regression_power(*args, **kwargs)
