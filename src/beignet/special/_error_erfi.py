from torch import Tensor

from ._error_erf import error_erf


def error_erfi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Imaginary error function.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
    """
    output = -1.0j * error_erf(1.0j * input)

    if out is not None:
        out.copy_(output)

        return out

    return output
