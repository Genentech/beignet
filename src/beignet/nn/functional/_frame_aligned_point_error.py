# ruff: noqa: E501

import torch
from torch import Tensor

import beignet


def frame_aligned_point_error(
    input: (Tensor, Tensor, Tensor),
    target: (Tensor, Tensor, Tensor),
    mask: (Tensor, Tensor),
    length_scale: float,
    pair_mask: Tensor | None = None,
    maximum: float | None = None,
    epsilon=1e-8,
) -> Tensor:
    """
    Parameters
    ----------
    input : (Tensor, Tensor, Tensor)
        A $3$-tuple of rotation matrices, translations, and positions. The
        rotation matrices must have the shape $(\\ldots, 3, 3)$, the
        translations must have the shape $(\\ldots, 3)$, and the positions must
        have the shape $(\\ldots, \text{points}, 3)$.

    target : (Tensor, Tensor, Tensor)
        A $3$-tuple of target rotation matrices, translations, and positions.
        The rotation matrices must have the shape $(\\ldots, 3, 3)$, the
        translations must have the shape $(\\ldots, 3)$, and the positions must
        have the shape $(\\ldots, \text{points}, 3)$.

    mask : (Tensor, Tensor)
        [*, N_frames] binary mask for the frames
        [..., points], position masks

    length_scale : float
        Length scale by which the loss is divided

    pair_mask : Tensor | None, optional
        [*,  N_frames, N_pts] mask to use for separating intra- from inter-chain losses.

    maximum : float | None, optional
        Cutoff above which distance errors are disregarded

    epsilon : float, optional
        Small value used to regularize denominators

    Returns
    -------
    output : Tensor
        Losses for each frame of shape $(\\ldots, 3)$.
    """
    output = torch.sqrt(torch.sum((beignet.apply_transform(input[1][..., None, :, :], beignet.invert_transform(input[0])) - beignet.apply_transform(target[1][..., None, :, :], beignet.invert_transform(target[0]), )) ** 2, dim=-1) + epsilon)  # fmt: off

    if maximum is not None:
        output = torch.clamp(output, 0, maximum)

    output = output / length_scale * mask[0][..., None] * mask[1][..., None, :]  # fmt: off

    if pair_mask is not None:
        output = torch.sum(output * pair_mask, dim=[-1, -2]) / (torch.sum(mask[0][..., None] * mask[1][..., None, :] * pair_mask, dim=[-2, -1]) + epsilon)  # fmt: off
    else:
        output = torch.sum((torch.sum(output, dim=-1) / (torch.sum(mask[0], dim=-1))[..., None] + epsilon), dim=-1) / (torch.sum(mask[1], dim=-1) + epsilon)  # fmt: off

    return output
