import torch
import torch._numpy._funcs_impl
from torch import Tensor


def _pad_along_axis(
    input: Tensor,
    padding=(0, 0),
    axis=0,
):
    input = torch.moveaxis(input, axis, 0)

    if padding[0] < 0:
        input = input[torch.abs(padding[0]) :]

        padding = (0, padding[1])

    if padding[1] < 0:
        input = input[: -torch.abs(padding[1])]

        padding = (padding[0], 0)

    npad = torch.tensor([(0, 0)] * input.ndim)

    npad[0] = padding

    output = torch._numpy._funcs_impl.pad(
        input,
        pad_width=npad,
        mode="constant",
        constant_values=0,
    )

    return torch.moveaxis(output, 0, axis)
