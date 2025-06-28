import torch
from torch import Tensor


def distogram_loss(
    input: Tensor,
    target: Tensor,
    mask: Tensor,
    start: float | Tensor = 2.3125,
    end: float | Tensor = 21.6875,
    steps: int = 64,
) -> Tensor:
    """

    Parameters
    ----------
    input
    target
    mask
    start
    end
    steps

    Returns
    -------

    """
    mask = mask[..., None] * mask[..., None, :]

    return torch.mean(
        torch.sum(
            (
                torch.sum(
                    -1
                    * torch.sum(
                        torch.nn.functional.one_hot(
                            torch.sum(
                                torch.sum(
                                    (target[..., None, :] - target[..., None, :, :])
                                    ** 2,
                                    dim=-1,
                                    keepdim=True,
                                )
                                > torch.linspace(start, end, steps - 1) ** 2,
                                dim=-1,
                            ),
                            steps,
                        )
                        * torch.nn.functional.log_softmax(input, dim=-1),
                        dim=-1,
                    )
                    * mask,
                    dim=-1,
                )
                / (torch.sum(mask, dim=[-1, -2]) + torch.finfo(input.dtype).eps)[
                    ..., None
                ]
            ),
            dim=-1,
        )
    )
