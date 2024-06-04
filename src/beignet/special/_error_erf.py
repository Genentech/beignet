from torch import Tensor

from ._error_erfc import error_erfc


def error_erf(input: Tensor) -> Tensor:
    r"""
    Error function.

    Parameters
    ----------
    input : Tensor

    Returns
    -------
    Tensor
    """
    return 1.0 - error_erfc(input)
