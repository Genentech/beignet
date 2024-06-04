from torch import Tensor

from ._error_erf import error_erf


def error_erfi(input: Tensor) -> Tensor:
    r"""
    Imaginary error function.

    Parameters
    ----------
    input : Tensor

    Returns
    -------
    Tensor
    """
    return -1.0j * error_erf(1.0j * input)
