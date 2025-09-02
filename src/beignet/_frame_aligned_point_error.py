import torch
from torch import Tensor

import beignet.rotations


def frame_aligned_point_error(
    rotations: Tensor,
    translations: Tensor,
    positions: Tensor,
    target_rotations: Tensor,
    target_translations: Tensor,
    target_positions: Tensor,
    *,
    clamp_distance: float = 10.0,
) -> Tensor:
    r"""
    Compute the Frame Aligned Point Error (FAPE) loss from AlphaFold.

    FAPE compares predicted atom positions to target positions under many
    different local frame alignments. For each alignment, it computes the distance
    between predicted and target atom positions after aligning the predicted frame
    to the corresponding target frame. This creates a strong bias for atoms
    to be correct relative to the local frame of each residue.

    Parameters
    ----------
    rotations : Tensor, shape=(..., N_frames, 3, 3)
        The predicted frame rotation matrices.

    translations : Tensor, shape=(..., N_frames, 3)
        The predicted frame translation vectors.

    positions : Tensor, shape=(..., N_atoms, 3)
        The predicted atom positions in global coordinates.

    target_rotations : Tensor, shape=(..., N_frames, 3, 3)
        The target frame rotation matrices.

    target_translations : Tensor, shape=(..., N_frames, 3)
        The target frame translation vectors.

    target_positions : Tensor, shape=(..., N_atoms, 3)
        The target atom positions in global coordinates.

    clamp_distance : float, default=10.0
        Maximum distance for clamping the L1 loss (in Angstroms).
        Set to None for unclamped loss.

    Returns
    -------
    loss : Tensor, shape=(...)
        The FAPE loss value.

    Notes
    -----
    The FAPE loss is computed as follows:
    1. For each frame k, align the predicted frame to the target frame
    2. Transform all atom positions into the aligned coordinate system
    3. Compute L1 distances between transformed predicted and target positions
    4. Clamp distances at the specified threshold
    5. Average over all frames and atoms

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> # Create sample data
    >>> batch_size, n_frames, n_atoms = 2, 4, 10
    >>> rotations = torch.randn(batch_size, n_frames, 3, 3)
    >>> translations = torch.randn(batch_size, n_frames, 3)
    >>> target_rotations = torch.randn(batch_size, n_frames, 3, 3)
    >>> target_translations = torch.randn(batch_size, n_frames, 3)
    >>> positions = torch.randn(batch_size, n_atoms, 3)
    >>> target_positions = torch.randn(batch_size, n_atoms, 3)
    >>> # Compute FAPE loss
    >>> loss = beignet.frame_aligned_point_error(rotations, translations, positions, target_rotations, target_translations, target_positions)
    >>> loss.shape
    torch.Size([2])
    """
    # Validate input shapes
    if rotations.shape != target_rotations.shape:
        raise ValueError(
            f"Rotation shapes must match: input {rotations.shape} "
            f"vs target {target_rotations.shape}"
        )

    if translations.shape != target_translations.shape:
        raise ValueError(
            f"Translation shapes must match: input {translations.shape} "
            f"vs target {target_translations.shape}"
        )

    if positions.shape != target_positions.shape:
        raise ValueError(
            f"Position shapes must match: input {positions.shape} "
            f"vs target {target_positions.shape}"
        )

    # Get dimensions
    n_frames = rotations.shape[-3]
    n_atoms = positions.shape[-2]

    # Compute alignment transformation: T_align = T_target^-1 * T_input
    # First, compute T_target^-1
    target_rot_inv = beignet.rotations.invert_rotation_matrix(target_rotations)
    target_trans_inv = -beignet.rotations.apply_rotation_matrix(
        target_translations, target_rot_inv
    )

    # Then compute T_align = T_target^-1 * T_input
    align_rot = beignet.rotations.compose_rotation_matrix(target_rot_inv, rotations)
    align_trans = target_trans_inv + beignet.rotations.apply_rotation_matrix(
        translations, target_rot_inv
    )

    # Expand positions for broadcasting over frames
    # (..., N_atoms, 3) -> (..., 1, N_atoms, 3)
    pos_expanded = torch.unsqueeze(positions, -3)
    target_pos_expanded = torch.unsqueeze(target_positions, -3)

    # FAPE Algorithm: For each frame k, align predicted frame to target frame,
    # then compare positions in the aligned coordinate system

    # Method: Transform both predicted and target positions using their respective frames,
    # then apply alignment to predicted positions to bring them to target frame coords

    # Transform predicted positions from global to predicted frame local, then align to target frame
    # Step 1: Global positions are already given, so we work with them directly
    # Step 2: Apply alignment transformation which brings predicted positions to target frame coords
    # Note: Using einsum here is more efficient than apply_rotation_matrix for this broadcasting pattern
    aligned_pred_pos = torch.einsum(
        "...ij,...kj->...ki", align_rot, pos_expanded
    ) + torch.unsqueeze(align_trans, -2)

    # Target positions are already in the right coordinate system (global)
    # but we need to view them from each frame's perspective
    aligned_target_pos = target_pos_expanded.expand_as(aligned_pred_pos)

    # Compute L1 distances between aligned positions
    # Shape: (..., N_frames, N_atoms, 3)
    position_diff = aligned_pred_pos - aligned_target_pos
    distances = torch.norm(position_diff, dim=-1)  # (..., N_frames, N_atoms)

    # Apply clamping if specified
    if clamp_distance is not None:
        distances = torch.clamp(distances, max=clamp_distance)

    # Average over frames and atoms
    total_loss = torch.sum(distances, dim=(-2, -1))  # Sum over frames and atoms
    normalization = n_frames * n_atoms

    eps = torch.finfo(total_loss.dtype).eps
    return total_loss / (normalization + eps)
