import numpy
import torch
from torch import Tensor


def polyvalfromroots(x, output, tensor=True):
    output = torch.ravel(output)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output.astype(numpy.float64)

    if isinstance(x, (tuple, list)):
        x = torch.tensor(x)

    if isinstance(x, Tensor):
        if tensor:
            shape = (1,) * x.ndim

            output = torch.reshape(output, [*output.shape, *shape])
        elif x.ndim >= output.ndim:
            raise ValueError

    return torch.prod(x - output, dim=0)
