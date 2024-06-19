import torch
from torch import Tensor

from .__as_series import _as_series


def _get_domain(input: Tensor) -> Tensor:
    (input,) = _as_series([input], trim=False)

    if torch.is_complex(input):
        return torch.tensor(
            [
                torch.complex(
                    torch.min(torch.real(input)),
                    torch.min(torch.imag(input)),
                ),
                torch.complex(
                    torch.max(torch.real(input)),
                    torch.max(torch.imag(input)),
                ),
            ],
        )
    else:
        return torch.tensor(
            [
                torch.min(input),
                torch.max(input),
            ],
        )
