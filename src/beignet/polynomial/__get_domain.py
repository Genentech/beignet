import torch
from torch import Tensor


def _get_domain(input: Tensor) -> Tensor:
    if torch.is_complex(input):
        output = torch.tensor(
            [
                torch.min(input.real) + 1j * torch.min(input.imag),
                torch.max(input.real) + 1j * torch.max(input.imag),
            ],
        )
        return output
    else:
        output = torch.tensor(
            [
                torch.min(input),
                torch.max(input),
            ],
        )

    return output
