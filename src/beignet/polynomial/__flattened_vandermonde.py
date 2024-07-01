import torch

from beignet.polynomial import _vandermonde


def _flattened_vandermonde(
    vandermonde_functions,
    points,
    degrees,
):
    vandermonde = _vandermonde(
        vandermonde_functions,
        points,
        degrees,
    )

    return torch.reshape(
        vandermonde,
        vandermonde.shape[: -len(degrees)] + (-1,),
    )
