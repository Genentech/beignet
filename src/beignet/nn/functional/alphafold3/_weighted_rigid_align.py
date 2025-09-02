import torch
from torch import Tensor


def weighted_rigid_align(
    input: Tensor,
    target: Tensor,
    weights: Tensor,
) -> Tensor:
    r"""
    Compute weighted rigid alignment between two sets of 3D points.

    This implements the weighted rigid align algorithm from AlphaFold 3, which
    finds the optimal rotation and translation to align input points to target
    points using per-point weights. This is a weighted generalization of the
    Kabsch algorithm.

    Parameters
    ----------
    input : Tensor, shape=(..., N, 3)
        Input 3D points to be aligned.

    target : Tensor, shape=(..., N, 3)
        Target 3D points to align to.

    weights : Tensor, shape=(..., N)
        Per-point weights for alignment. Higher weights give more importance
        to the corresponding points in the alignment.

    Returns
    -------
    aligned : Tensor, shape=(..., N, 3)
        Input points after optimal rigid alignment to target points.
        Gradients are stopped with respect to the aligned coordinates.

    Notes
    -----
    The weighted rigid alignment algorithm:

    1. Compute weighted centroids of both point sets
    2. Center both point sets by subtracting their weighted centroids
    3. Form the weighted cross-covariance matrix H = Σ wᵢ target_iᵀ input_i
    4. Compute SVD: H = UΣVᵀ
    5. Compute rotation matrix: R = UVᵀ
    6. Handle reflection case: if det(R) < 0, flip the last column of U
    7. Apply alignment: aligned = R @ input_centered + target_centroid

    The `stop_gradient` operation prevents gradients from flowing through
    the aligned coordinates, which is typical in structure prediction models
    where the alignment is used for loss computation but shouldn't contribute
    to gradient updates of the aligned positions themselves.

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> batch_size, n_points = 2, 10
    >>> input = torch.randn(batch_size, n_points, 3)
    >>> target = torch.randn(batch_size, n_points, 3)
    >>> weights = torch.ones(batch_size, n_points)  # Equal weights
    >>> aligned = beignet.weighted_rigid_align(input, target, weights)
    >>> aligned.shape
    torch.Size([2, 10, 3])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 28: Weighted Rigid Align
    """
    # Validate input shapes
    if input.shape != target.shape:
        raise ValueError(
            f"Input and target shapes must match: "
            f"input {input.shape} vs target {target.shape}"
        )

    if input.shape[-1] != 3:
        raise ValueError(f"Points must be 3D: got shape {input.shape}")

    if weights.shape != input.shape[:-1]:
        raise ValueError(
            f"Weights shape must match points shape except last dimension: "
            f"weights {weights.shape} vs points {input.shape[:-1]}"
        )

    # Expand weights for broadcasting: (..., N) -> (..., N, 1)
    weights_expanded = torch.unsqueeze(weights, -1)

    # Compute weighted centroids (steps 1-2)
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)  # (..., 1)
    eps = torch.finfo(input.dtype).eps

    input_centroid = torch.sum(weights_expanded * input, dim=-2, keepdim=True) / (
        weight_sum.unsqueeze(-1) + eps
    )
    target_centroid = torch.sum(weights_expanded * target, dim=-2, keepdim=True) / (
        weight_sum.unsqueeze(-1) + eps
    )

    # Center the point sets (steps 3-4)
    input_centered = input - input_centroid
    target_centered = target - target_centroid

    # Form weighted cross-covariance matrix H = Σ wᵢ target_iᵀ input_i (step 5)
    # H has shape (..., 3, 3)
    weighted_target = weights_expanded * target_centered  # (..., N, 3)
    H = torch.matmul(
        torch.transpose(weighted_target, -2, -1),  # (..., 3, N)
        input_centered,  # (..., N, 3)
    )  # (..., 3, 3)

    # Compute SVD: H = UΣVᵀ (step 5)
    U, S, Vt = torch.linalg.svd(H)

    # Compute rotation matrix R = UVᵀ (step 6)
    R = torch.matmul(U, Vt)

    # Handle reflection case (steps 7-10)
    # If det(R) < 0, we have a reflection, so flip the last column of U
    det_R = torch.linalg.det(R)
    reflection_mask = det_R < 0

    # Create F matrix for reflection correction
    F = torch.eye(3, dtype=input.dtype, device=input.device)
    F = F.expand_as(U)
    F = F.clone()
    F[..., 2, 2] = torch.where(reflection_mask, -1.0, 1.0)

    # Apply reflection correction: R = UFVᵀ
    R = torch.where(
        reflection_mask.unsqueeze(-1).unsqueeze(-1),
        torch.matmul(torch.matmul(U, F), Vt),
        R,
    )

    # Apply alignment (step 11): aligned = R @ input_centered + target_centroid
    aligned = torch.matmul(R, torch.transpose(input_centered, -2, -1))
    aligned = torch.transpose(aligned, -2, -1) + target_centroid

    # Stop gradients (step 12)
    aligned = aligned.detach()

    return aligned
