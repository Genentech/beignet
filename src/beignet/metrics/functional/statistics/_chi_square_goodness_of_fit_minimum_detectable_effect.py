"""chi square goodness of fit minimum detectable effect functional metric."""

import beignet.statistics


def chi_square_goodness_of_fit_minimum_detectable_effect(*args, **kwargs):
    """
    Compute chi_square_goodness_of_fit_minimum_detectable_effect.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.chi_square_goodness_of_fit_minimum_detectable_effect(
        *args, **kwargs
    )
