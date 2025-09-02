import torch
from torch import Tensor


def smooth_local_distance_difference_test(
    input: Tensor,
    target: Tensor,
    *,
    cutoff_radius: float = 15.0,
    tolerance_thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    smoothing_factor: float = 1.0,
) -> Tensor:
    r"""
    Compute the smooth Local Distance Difference Test (LDDT) loss.

    The smooth LDDT loss is a differentiable version of the LDDT metric used for
    protein structure assessment. It measures how well local distance relationships
    are preserved between predicted and target structures within a cutoff radius.

    Parameters
    ----------
    input : Tensor, shape=(..., N, 3)
        Predicted atom positions in 3D space.

    target : Tensor, shape=(..., N, 3)
        Target (reference) atom positions in 3D space.

    cutoff_radius : float, default=15.0
        Maximum distance (in Angstroms) for considering atom pairs in LDDT calculation.
        Only pairs within this distance in the target structure are evaluated.

    tolerance_thresholds : tuple[float, ...], default=(0.5, 1.0, 2.0, 4.0)
        Distance difference thresholds (in Angstroms) for LDDT scoring.
        Standard LDDT uses these four thresholds to define accuracy levels.

    smoothing_factor : float, default=1.0
        Controls the smoothness of the sigmoid approximations. Higher values
        make the function more similar to discrete LDDT but less smooth.

    Returns
    -------
    loss : Tensor, shape=(...)
        The smooth LDDT loss value (1 - smooth_LDDT_score).
        Lower values indicate better structural agreement.

    Notes
    -----
    The smooth LDDT loss is computed as follows:

    1. For each pair of atoms (i, j) in the target structure:
       - Check if the distance d_target(i,j) ≤ cutoff_radius
       - If yes, include this pair in the evaluation set

    2. For each included pair, compute the distance difference:
       - Δd = |d_predicted(i,j) - d_target(i,j)|

    3. For each tolerance threshold t, compute a smooth score:
       - smooth_score_t = sigmoid(smoothing_factor * (t - Δd))
       - This approximates the step function: 1 if Δd ≤ t, 0 otherwise

    4. Average over all thresholds to get the smooth LDDT score for each pair

    5. Average over all valid pairs to get the final smooth LDDT score

    6. Return 1 - smooth_LDDT_score as the loss (so minimizing loss maximizes LDDT)

    The smooth LDDT score ranges from 0 to 1, where 1 indicates perfect
    structural agreement. The loss therefore ranges from 0 to 1, where 0
    indicates perfect agreement.

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> batch_size, n_atoms = 2, 10
    >>> input = torch.randn(batch_size, n_atoms, 3)
    >>> target = torch.randn(batch_size, n_atoms, 3)
    >>> loss = beignet.smooth_local_distance_difference_test(input, target)
    >>> loss.shape
    torch.Size([2])
    """
    # Validate input shapes
    if input.shape != target.shape:
        raise ValueError(
            f"Position shapes must match: predicted {input.shape} "
            f"vs target {target.shape}"
        )

    if input.shape[-1] != 3:
        raise ValueError(f"Positions must have 3 coordinates: got shape {input.shape}")

    # Get dimensions
    n_atoms = input.shape[-2]

    # Compute pairwise distances in target and predicted structures
    # input: (..., N, 3) -> (..., N, 1, 3) and (..., 1, N, 3)
    pred_i = torch.unsqueeze(input, -2)  # (..., N, 1, 3)
    pred_j = torch.unsqueeze(input, -3)  # (..., 1, N, 3)

    target_i = torch.unsqueeze(target, -2)  # (..., N, 1, 3)
    target_j = torch.unsqueeze(target, -3)  # (..., 1, N, 3)

    # Compute distances
    eps = torch.finfo(pred_i.dtype).eps
    pred_distances = torch.sqrt(
        torch.sum((pred_i - pred_j) ** 2, dim=-1) + eps
    )  # (..., N, N)
    target_distances = torch.sqrt(
        torch.sum((target_i - target_j) ** 2, dim=-1) + eps
    )  # (..., N, N)

    # Create mask for valid pairs:
    # 1. Not diagonal (i != j)
    # 2. Within cutoff radius in target structure
    diagonal_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=input.device)
    cutoff_mask = target_distances <= cutoff_radius
    valid_mask = diagonal_mask & cutoff_mask  # (..., N, N)

    # Compute distance differences for valid pairs
    distance_diff = torch.abs(pred_distances - target_distances)  # (..., N, N)

    # Apply mask to get only valid pairs
    masked_distance_diff = torch.where(
        valid_mask, distance_diff, torch.zeros_like(distance_diff)
    )

    # Compute smooth LDDT scores for each tolerance threshold
    smooth_scores = []
    for threshold in tolerance_thresholds:
        # Use sigmoid as smooth approximation of step function
        # sigmoid(smoothing_factor * (threshold - distance_diff)) ≈ 1 if distance_diff <= threshold
        smooth_score = torch.sigmoid(
            smoothing_factor * (threshold - masked_distance_diff)
        )
        smooth_scores.append(smooth_score)

    # Stack and average over thresholds: (..., N, N, num_thresholds) -> (..., N, N)
    smooth_scores_tensor = torch.stack(
        smooth_scores, dim=-1
    )  # (..., N, N, num_thresholds)
    avg_smooth_score_per_pair = torch.mean(smooth_scores_tensor, dim=-1)  # (..., N, N)

    # Apply valid mask and average over valid pairs
    masked_scores = torch.where(
        valid_mask,
        avg_smooth_score_per_pair,
        torch.zeros_like(avg_smooth_score_per_pair),
    )

    # Count valid pairs for normalization
    num_valid_pairs = torch.sum(valid_mask.float(), dim=(-2, -1))  # (...)

    # Compute mean smooth LDDT score over valid pairs
    total_score = torch.sum(masked_scores, dim=(-2, -1))  # (...)
    smooth_lddt_score = total_score / (num_valid_pairs + eps)

    # Return loss as 1 - smooth_LDDT_score
    # This ensures that minimizing the loss maximizes the LDDT score
    loss = 1.0 - smooth_lddt_score

    return loss
