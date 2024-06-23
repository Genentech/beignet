import torch
from torch import Tensor


def chebpts2(input: Tensor) -> Tensor:
    _npts = int(input)

    if _npts != input:
        raise ValueError

    if _npts < 2:
        raise ValueError

    output = torch.linspace(-torch.pi, 0, _npts)

    return torch.cos(output)
