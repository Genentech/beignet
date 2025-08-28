"""multivariable linear regression sample size functional metric."""

import beignet.statistics


def multivariable_linear_regression_sample_size(*args, **kwargs):
    """
    Compute multivariable_linear_regression_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.multivariable_linear_regression_sample_size(
        *args, **kwargs
    )
