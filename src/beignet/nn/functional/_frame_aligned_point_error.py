from typing import Tuple

import torch
from torch import Tensor

import beignet


def frame_aligned_point_error(
    input: Tuple[Tuple[Tensor, Tensor], Tensor],
    target: Tuple[Tuple[Tensor, Tensor], Tensor],
    mask: Tuple[Tensor, Tensor],
    z: float,
    mp: Tensor | None = None,
    wk: float | None = None,
) -> Tensor:
    r"""
    Score a set of predicted atom coordinates, $\left\{\vec{x}_{j}\right\}$,
    under a set of predicted local frames, $\left\{T_{i}\right\}$, against the
    corresponding target atom coordinates,
    $\left\{\vec{x}_{i}^{\mathrm{True}}\right\}$, and target local frames,
    $\left\{T_{i}^{\mathrm{True}}\right\}$. All atoms in all backbone and side
    chain frames are scored.

    Additionally, a cheaper version (scoring only all $C_\alpha$ atoms in all
    backbone frames) is used as an auxiliary loss in every layer of the
    AlphaFold structure module.

    In order to formulate the loss the atom position $\vec{x}_{j}$ is computed
    relative to frame $T_{i}$ and the location of the corresponding true atom
    position $\vec{x}_{j}^{\mathrm{True}}$ relative to the true frame
    $T_{i}^{\mathrm{True}}$. The deviation is computed as a robust L2 norm
    ($\epsilon$ is a small constant added to ensure that gradients are
    numerically well behaved for small differences. The exact value of this
    constant does not matter, as long as it is small enough.

    The $N_{\mathrm{frames}} \times N_{\mathrm{atoms}}$ deviations are
    penalized with a clamped L1 loss with a length scale, $Z = 10\text{\r{A}}$,
    to make the loss unitless.

    Parameters
    ----------
    input : Tensor, (Tensor, Tensor)
        A pair of predicted atom coordinates, $\left\{\vec{x}_{j}\right\}$, and
        predicted local frames, $\left\{T_{i}\right\}$. A frame is represented
        as a pair of rotation matrices and corresponding translations. The
        predicted atom positions must have the shape
        $(\\ldots, \text{points}, 3)$, the rotation matrices must have the
        shape $(\\ldots, 3, 3)$, and the translations must have the shape
        $(\\ldots, 3)$.

    target : Tensor, (Tensor, Tensor)
        A pair of target atom coordinates,
        $\left\{\vec{x}_{i}^{\mathrm{True}}\right\}$, and target local frames,
        $\left\{T_{i}^{\mathrm{True}}\right\}$. A frame is represented as a
        pair of rotation matrices and corresponding translations. The predicted
        atom positions must have the shape $(\\ldots, \text{points}, 3)$, the
        rotation matrices must have the shape $(\\ldots, 3, 3)$, and the
        translations must have the shape $(\\ldots, 3)$.

    mask : (Tensor, Tensor)
        [*, N_frames] binary mask for the frames
        [..., points], position masks

    z : float
        Length scale by which the loss is divided

    mp : Tensor | None, optional
        [*,  N_frames, N_pts] mask to use for separating intra-chain from
        inter-chain losses.

    wk : float | None, optional
        Cutoff above which distance errors are disregarded

    Returns
    -------
    output : Tensor
        Losses for each frame of shape $(\\ldots, 3)$.
    """
    transform, input = input

    target_transform, target = target

    epsilon = torch.finfo(input.dtype).eps

    input = beignet.apply_transform(
        input[..., None, :, :],
        beignet.invert_transform(
            transform,
        ),
    )

    target = beignet.apply_transform(
        target[..., None, :, :],
        beignet.invert_transform(
            target_transform,
        ),
    )

    output = torch.sqrt(torch.sum((input - target) ** 2, dim=-1) + epsilon)

    if wk is not None:
        output = torch.clamp(output, 0, wk)

    output = output / z * mask[0][..., None] * mask[1][..., None, :]

    if mp is not None:
        output = torch.sum(output * mp, dim=[-1, -2]) / (
            torch.sum(mask[0][..., None] * mask[1][..., None, :] * mp, dim=[-2, -1])
            + epsilon
        )
    else:
        output = torch.sum(
            torch.sum(output, dim=-1)
            / (torch.sum(mask[0], dim=-1)[..., None] + epsilon),
            dim=-1,
        ) / (torch.sum(mask[1], dim=-1) + epsilon)

    return output
