"""mann whitney u test minimum detectable effect functional metric."""

import beignet.statistics


def mann_whitney_u_test_minimum_detectable_effect(*args, **kwargs):
    """
    Compute mann_whitney_u_test_minimum_detectable_effect.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.mann_whitney_u_test_minimum_detectable_effect(
        *args,
        **kwargs,
    )
