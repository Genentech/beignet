from __future__ import annotations

import torch
from torch import Tensor

from ._rigid import Rigid


def fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    pred_positions: Tensor,
    target_positions: Tensor,
    *,
    frames_mask: Tensor | None = None,
    positions_mask: Tensor | None = None,
    length_scale: float = 10.0,
    clamp_distance: float | None = 10.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Compute the AlphaFold Frame Aligned Point Error (FAPE).

    The FAPE compares predicted points expressed in the coordinates of a
    predicted frame to target points expressed in the coordinates of the
    corresponding target frame. Distances are optionally clamped and scaled
    by a characteristic length so the loss is typically in [0, 1].

    Parameters
    ----------
    pred_frames
        Predicted rigid transforms with shape ``[..., N_frame]``.
    target_frames
        Target/ground-truth rigid transforms with shape broadcastable to
        ``pred_frames``.
    pred_positions
        Predicted points with shape ``[..., N_point, 3]``.
    target_positions
        Target/ground-truth points with shape broadcastable to
        ``pred_positions``.
    frames_mask
        Optional mask for frames with shape ``[..., N_frame]`` (boolean).
    positions_mask
        Optional mask for points with shape ``[..., N_point]`` (boolean).
    length_scale
        Characteristic length to scale the error. In AlphaFold this is
        typically ``10.0`` for backbone/sidechain FAPE.
    clamp_distance
        If provided, clamp the per-pair distance to this maximum before
        scaling. Set to ``None`` to disable clamping.
    eps
        Small constant for numerical stability.
    reduction
        One of ``"mean"``, ``"sum"``, or ``"none"`` specifying how to
        reduce over frames/points. ``"mean"`` divides by the number of
        valid pairs given by the masks.

    Returns
    -------
    Tensor
        If ``reduction != 'none'``, returns a scalar tensor. Otherwise,
        returns the unreduced per-pair FAPE with shape
        ``[..., N_frame, N_point]``.

    Notes
    -----
    - Let ``T_i`` be a frame and ``x_j`` a point. FAPE measures
      ``|| T_i^{-1}(x_j^pred) - (T_i^{*})^{-1}(x_j^{*}) ||``.
    - When masks are provided, a pair contributes if both the frame and the
      point are valid.
    """

    # Express points in local coordinates of their corresponding frames
    # shapes broadcast to [..., N_frame, N_point, 3]
    pred_local = pred_frames.inverse()(pred_positions)
    target_local = target_frames.inverse()(target_positions)

    # Euclidean distance in local coordinates
    d = torch.linalg.norm(pred_local - target_local, dim=-1)

    if clamp_distance is not None:
        d = torch.clamp(d, max=float(clamp_distance))

    loss_per_pair = d / float(length_scale)

    if frames_mask is not None:
        frames_mask = frames_mask.to(dtype=loss_per_pair.dtype)
        # [..., N_frame, 1]
        frames_mask = frames_mask.unsqueeze(-1)
        loss_per_pair = loss_per_pair * frames_mask

    if positions_mask is not None:
        positions_mask = positions_mask.to(dtype=loss_per_pair.dtype)
        # [..., 1, N_point]
        positions_mask = positions_mask.unsqueeze(-2)
        loss_per_pair = loss_per_pair * positions_mask

    if reduction == "none":
        return loss_per_pair

    if reduction not in {"mean", "sum"}:
        raise ValueError(
            f"reduction must be one of {{'mean', 'sum', 'none'}}, got: {reduction!r}"
        )

    if frames_mask is None and positions_mask is None:
        if reduction == "sum":
            return loss_per_pair.sum()
        return loss_per_pair.mean()

    # Compute denominator = number of valid pairs
    denom_mask = torch.ones_like(loss_per_pair)
    if frames_mask is not None:
        denom_mask = denom_mask * frames_mask
    if positions_mask is not None:
        denom_mask = denom_mask * positions_mask

    total = loss_per_pair.sum()
    denom = denom_mask.sum().clamp_min(eps)

    if reduction == "sum":
        return total
    return total / denom
