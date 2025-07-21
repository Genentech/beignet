from typing import Optional

import torch
from torch import Tensor


def local_distance_difference_test(
    predicted_coords: Tensor,
    reference_coords: Tensor,
    atom_mask: Optional[Tensor] = None,
    cutoff: float = 15.0,
    thresholds: Optional[list[float]] = None,
    per_atom: bool = False,
) -> Tensor:
    r"""
    Compute the Local Distance Difference Test (LDDT) score for protein structure evaluation.

    LDDT is a superposition-free score that evaluates how well the local distances
    between atoms are preserved in the predicted structure compared to the reference.
    It measures the fraction of atom pairs within a cutoff distance that have their
    distances preserved within specified thresholds.

    Parameters
    ----------
    predicted_coords : Tensor, shape=(..., N, 3)
        Predicted atom coordinates.
    reference_coords : Tensor, shape=(..., N, 3)
        Reference (true) atom coordinates.
    atom_mask : Tensor, shape=(..., N), optional
        Binary mask indicating valid atoms (1 for valid, 0 for invalid).
        If None, all atoms are considered valid.
    cutoff : float, default=15.0
        Maximum distance cutoff in Angstroms for considering atom pairs.
    thresholds : List[float], optional
        Distance difference thresholds in Angstroms. The score is averaged
        over these thresholds. Default: [0.5, 1.0, 2.0, 4.0]
    per_atom : bool, default=False
        If True, return per-atom LDDT scores. If False, return global average.

    Returns
    -------
    lddt_score : Tensor
        LDDT scores in range [0, 1]. Shape is (..., N) if per_atom=True,
        otherwise (...,).

    Examples
    --------
    >>> batch_size, n_atoms = 2, 100
    >>> predicted = torch.randn(batch_size, n_atoms, 3)
    >>> reference = predicted + torch.randn_like(predicted) * 0.1
    >>> mask = torch.ones(batch_size, n_atoms)
    >>> lddt = local_distance_difference_test(predicted, reference, mask)
    >>> lddt.shape
    torch.Size([2])
    >>> assert torch.all(lddt >= 0) and torch.all(lddt <= 1)
    """
    # Validate inputs
    if predicted_coords.shape != reference_coords.shape:
        raise ValueError(
            f"Predicted and reference coordinates must have the same shape, "
            f"got {predicted_coords.shape} and {reference_coords.shape}"
        )

    if predicted_coords.shape[-1] != 3:
        raise ValueError(
            f"Coordinates must have 3 dimensions (x, y, z), got {predicted_coords.shape[-1]}"
        )

    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = [0.5, 1.0, 2.0, 4.0]

    if any(t <= 0 for t in thresholds):
        raise ValueError("Thresholds must be positive")

    # Get dimensions
    *batch_dims, n_atoms, _ = predicted_coords.shape
    device = predicted_coords.device

    # Create atom mask if not provided
    if atom_mask is None:
        atom_mask = torch.ones(*batch_dims, n_atoms, dtype=torch.bool, device=device)
    else:
        atom_mask = atom_mask.bool()

    # Compute pairwise distances for both predicted and reference
    # Shape: (..., N, N)
    # Using explicit computation for better gradient support
    pred_diff = predicted_coords.unsqueeze(-2) - predicted_coords.unsqueeze(-3)
    pred_distances = torch.norm(pred_diff, p=2, dim=-1)

    ref_diff = reference_coords.unsqueeze(-2) - reference_coords.unsqueeze(-3)
    ref_distances = torch.norm(ref_diff, p=2, dim=-1)

    # Create pair mask: both atoms must be valid and within cutoff in reference
    # Shape: (..., N, N)
    pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)  # Both atoms valid
    pair_mask = pair_mask & (ref_distances < cutoff)  # Within cutoff
    # Exclude self-interactions
    pair_mask = pair_mask & ~torch.eye(n_atoms, dtype=torch.bool, device=device)

    # Compute distance differences
    distance_diff = torch.abs(pred_distances - ref_distances)

    # For each threshold, compute fraction of preserved distances
    threshold_scores = []
    for threshold in thresholds:
        # Count pairs where distance is preserved within threshold
        preserved = (distance_diff < threshold).float() * pair_mask.float()

        if per_atom:
            # Per-atom score: average over all pairs involving each atom
            # Shape: (..., N)
            numerator = preserved.sum(dim=-1).float()
            denominator = pair_mask.sum(dim=-1).float()
            # Avoid division by zero with smooth approximation
            atom_scores = numerator / (denominator + 1e-10)
            threshold_scores.append(atom_scores)
        else:
            # Global score: average over all valid pairs
            # Shape: (...)
            numerator = preserved.sum(dim=[-2, -1]).float()
            denominator = pair_mask.sum(dim=[-2, -1]).float()
            # Avoid division by zero with smooth approximation
            global_scores = numerator / (denominator + 1e-10)
            threshold_scores.append(global_scores)

    # Average scores across thresholds
    lddt_score = torch.stack(threshold_scores, dim=-1).mean(dim=-1)

    return lddt_score
