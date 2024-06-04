import math

import torch
from torch import Tensor

from ._error_erfi import error_erfi


def dawson_integral_f(z: Tensor) -> Tensor:
    return math.sqrt(torch.pi) / 2.0 * torch.exp(-torch.square(z)) * error_erfi(z)
