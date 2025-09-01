from torch import Tensor

from ._error_erfc import error_erfc


def error_erf(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Error function.

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
    output = 1.0 - error_erfc(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
