import math

import torch
from torch import Tensor

from ._error_erfi import error_erfi


def dawson_integral_f(input: Tensor) -> Tensor:
    return math.sqrt(torch.pi) / 2.0 * torch.exp(-(input**2)) * error_erfi(input)
