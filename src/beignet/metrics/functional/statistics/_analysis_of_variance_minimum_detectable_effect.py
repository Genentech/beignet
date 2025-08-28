"""analysis of variance minimum detectable effect functional metric."""

import beignet.statistics


def analysis_of_variance_minimum_detectable_effect(*args, **kwargs):
    """
    Compute analysis_of_variance_minimum_detectable_effect.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.analysis_of_variance_minimum_detectable_effect(
        *args,
        **kwargs,
    )
