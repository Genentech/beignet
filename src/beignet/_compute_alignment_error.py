import torch
from torch import Tensor

from ._express_coordinates_in_frame import express_coordinates_in_frame


def compute_alignment_error(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    predicted_frames: Tensor,
    target_frames: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    r"""
    Compute alignment error between predicted and target coordinates in local frames.

    This implements Algorithm 30 from AlphaFold 3, which computes the alignment
    error by expressing both predicted and target coordinates in their respective
    local frames and measuring the distance between them.

    Parameters
    ----------
    predicted_coordinates : Tensor, shape=(..., N, 3)
        Predicted 3D coordinates.

    target_coordinates : Tensor, shape=(..., N, 3)
        Target (ground truth) 3D coordinates.

    predicted_frames : Tensor, shape=(..., N, 3, 3)
        Predicted frame definitions. Each frame[..., i, :, :] defines a local
        coordinate system using three frame atoms for coordinate i.

    target_frames : Tensor, shape=(..., N, 3, 3)
        Target frame definitions. Each frame[..., i, :, :] defines a local
        coordinate system using three frame atoms for coordinate i.

    epsilon : float, default=1e-8
        Small value for numerical stability in the distance computation.

    Returns
    -------
    errors : Tensor, shape=(..., N)
        Alignment errors for each coordinate. Each error[..., i] represents
        the distance between the predicted and target coordinates when expressed
        in their respective local frames.

    Notes
    -----
    The algorithm follows these steps:

    1. For each coordinate i and frame j:
       - Express predicted coordinate x_i in predicted frame Φ_j
       - Express target coordinate x_i^true in target frame Φ_j^true
    2. Compute distance: e_ij = ||x̃_ij - x̃_ij^true||₂ + ε
    3. Return the set of all errors {e_ij}

    This measures how well the predicted coordinates align with the target
    coordinates when viewed from their respective local frame perspectives.
    The epsilon parameter prevents numerical issues when coordinates are
    perfectly aligned.

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> batch_size, n_coords = 2, 10
    >>> # Predicted and target coordinates
    >>> pred_coords = torch.randn(batch_size, n_coords, 3)
    >>> target_coords = torch.randn(batch_size, n_coords, 3)
    >>> # Frame definitions
    >>> pred_frames = torch.randn(batch_size, n_coords, 3, 3)
    >>> target_frames = torch.randn(batch_size, n_coords, 3, 3)
    >>> # Compute alignment errors
    >>> errors = beignet.compute_alignment_error(
    ...     pred_coords, target_coords, pred_frames, target_frames
    ... )
    >>> errors.shape
    torch.Size([2, 10])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 30: Compute alignment error
    """
    # Validate input shapes
    if predicted_coordinates.shape != target_coordinates.shape:
        raise ValueError(
            f"Predicted and target coordinates must have same shape: "
            f"{predicted_coordinates.shape} vs {target_coordinates.shape}"
        )

    if predicted_frames.shape != target_frames.shape:
        raise ValueError(
            f"Predicted and target frames must have same shape: "
            f"{predicted_frames.shape} vs {target_frames.shape}"
        )

    if predicted_coordinates.shape[-1] != 3:
        raise ValueError(
            f"Coordinates must be 3D: got shape {predicted_coordinates.shape}"
        )

    if predicted_frames.shape[-2:] != (3, 3):
        raise ValueError(
            f"Frames must have shape (..., N, 3, 3): got shape {predicted_frames.shape}"
        )

    batch_shape = predicted_coordinates.shape[:-2]
    n_coords = predicted_coordinates.shape[-2]

    if predicted_frames.shape[:-2] != batch_shape + (n_coords,):
        raise ValueError(
            f"Frame shape must match coordinate shape: "
            f"expected {batch_shape + (n_coords, 3, 3)}, got {predicted_frames.shape}"
        )

    # Get dimensions
    n_coords = predicted_coordinates.shape[-2]

    # Compute alignment errors for all coordinate-frame pairs
    errors = []

    for i in range(n_coords):  # For each coordinate
        coord_errors = []

        for j in range(n_coords):  # For each frame
            # Express predicted coordinate i in predicted frame j (step 1)
            pred_coord_i = predicted_coordinates[..., i, :]  # (..., 3)
            pred_frame_j = predicted_frames[..., j, :, :]  # (..., 3, 3)

            pred_in_frame = express_coordinates_in_frame(
                pred_coord_i, pred_frame_j
            )  # (..., 3)

            # Express target coordinate i in target frame j (step 2)
            target_coord_i = target_coordinates[..., i, :]  # (..., 3)
            target_frame_j = target_frames[..., j, :, :]  # (..., 3, 3)

            target_in_frame = express_coordinates_in_frame(
                target_coord_i, target_frame_j
            )  # (..., 3)

            # Compute distance between frame-expressed coordinates (step 3)
            diff = pred_in_frame - target_in_frame  # (..., 3)
            distance = torch.sqrt(torch.sum(diff**2, dim=-1) + epsilon)  # (...,)

            coord_errors.append(distance)

        # Stack errors for all frames for this coordinate
        coord_error_tensor = torch.stack(coord_errors, dim=-1)  # (..., n_coords)
        errors.append(coord_error_tensor)

    # Stack errors for all coordinates
    # Result shape: (..., n_coords, n_coords) where errors[..., i, j] = e_ij
    all_errors = torch.stack(errors, dim=-2)  # (..., n_coords, n_coords)

    return all_errors
