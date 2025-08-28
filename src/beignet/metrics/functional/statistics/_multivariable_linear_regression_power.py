"""multivariable linear regression power functional metric."""

import beignet.statistics


def multivariable_linear_regression_power(*args, **kwargs):
    """
    Compute multivariable_linear_regression_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.multivariable_linear_regression_power(*args, **kwargs)
