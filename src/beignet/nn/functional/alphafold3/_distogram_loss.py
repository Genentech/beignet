import torch
from torch import Tensor


def distogram_loss(
    input: Tensor,
    target: Tensor,
    *,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    num_bins: int = 64,
) -> Tensor:
    r"""
    Compute the AlphaFold 3 distogram loss.

    The distogram loss predicts probability distributions over distance bins for
    pairs of atoms. It uses cross-entropy loss between predicted distance
    distributions and target distance distributions computed from true atom
    target.

    Parameters
    ----------
    input : Tensor, shape=(..., N, N, num_bins)
        Predicted logits for distance distribution over bins for each pair of atoms.
        The logits should be symmetric (logits[..., i, j, :] ≈ logits[..., j, i, :]).

    target : Tensor, shape=(..., N, 3)
        True atom target in 3D space (in Angstroms).

    min_distance : float, default=2.0
        Minimum distance for the first bin (in Angstroms).

    max_distance : float, default=22.0
        Maximum distance for the last bin (in Angstroms).

    num_bins : int, default=64
        Number of distance bins to use.

    Returns
    -------
    loss : Tensor, shape=(...)
        The distogram cross-entropy loss value.

    Notes
    -----
    The distogram loss is computed as follows:
    1. Compute true pairwise distances from atom target
    2. Convert distances to target probability distributions over bins
    3. Apply softmax to predicted logits to get predicted distributions
    4. Compute cross-entropy loss between predicted and target distributions
    5. Average over all atom pairs (excluding self-pairs)

    The distance bins have equal width from min_distance to max_distance.
    Distances below min_distance are assigned to the first bin, and distances
    above max_distance are assigned to the last bin.

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> batch_size, n_atoms, num_bins = 2, 10, 64
    >>> input = torch.randn(batch_size, n_atoms, n_atoms, num_bins)
    >>> target = torch.randn(batch_size, n_atoms, 3)
    >>> loss = beignet.distogram_loss(input, target)
    >>> loss.shape
    torch.Size([2])
    """
    # Validate input shapes
    if input.shape[-3] != input.shape[-2]:
        raise ValueError(
            f"Predicted logits must be square in the last two dimensions: "
            f"got shape {input.shape}"
        )

    if input.shape[-1] != num_bins:
        raise ValueError(
            f"Predicted logits last dimension must match num_bins: "
            f"got {input.shape[-1]}, expected {num_bins}"
        )

    if target.shape[-2] != input.shape[-3]:
        raise ValueError(
            f"Number of atoms must match between target and input: "
            f"target shape {target.shape}, logits shape {input.shape}"
        )

    if target.shape[-1] != 3:
        raise ValueError(f"Positions must have 3 coordinates: got shape {target.shape}")

    # Get dimensions
    n_atoms = target.shape[-2]

    # Compute pairwise distances
    # target: (..., N, 3) -> (..., N, 1, 3) and (..., 1, N, 3)
    pos_i = torch.unsqueeze(target, -2)  # (..., N, 1, 3)
    pos_j = torch.unsqueeze(target, -3)  # (..., 1, N, 3)

    # Compute squared distances and take square root
    squared_diffs = torch.sum((pos_i - pos_j) ** 2, dim=-1)  # (..., N, N)
    eps = torch.finfo(squared_diffs.dtype).eps
    distances = torch.sqrt(squared_diffs + eps)  # Add eps for numerical stability

    # Create distance bins
    bin_width = (max_distance - min_distance) / (num_bins - 1)

    # Convert distances to soft target distributions to maintain gradients
    # Create bin centers
    bin_centers = torch.linspace(
        min_distance,
        max_distance,
        num_bins,
        dtype=distances.dtype,
        device=distances.device,
    )

    # Clamp distances to [min_distance, max_distance] range
    clamped_distances = torch.clamp(distances, min_distance, max_distance)

    # Create soft target distributions using Gaussian kernels around bin centers
    # This maintains differentiability w.r.t. target
    sigma = bin_width / 2.0  # Standard deviation for soft binning

    # Expand dimensions for broadcasting: (..., N, N, 1) and (num_bins,)
    distances_expanded = torch.unsqueeze(clamped_distances, -1)  # (..., N, N, 1)

    # Compute Gaussian weights for each bin
    diff_squared = (distances_expanded - bin_centers) ** 2
    target_probs = torch.exp(-diff_squared / (2 * sigma**2))

    # Normalize to get probability distribution
    target_probs = target_probs / (torch.sum(target_probs, dim=-1, keepdim=True) + eps)

    # Apply softmax to predicted logits to get probability distributions
    predicted_probs = torch.softmax(input, dim=-1)

    # Compute cross-entropy loss
    # CE = -sum(target * log(predicted))
    log_predicted_probs = torch.log(predicted_probs + eps)
    ce_loss = -torch.sum(target_probs * log_predicted_probs, dim=-1)  # (..., N, N)

    # Exclude self-pairs (diagonal elements) and average
    # Create mask to exclude diagonal
    mask = ~torch.eye(n_atoms, dtype=torch.bool, device=distances.device)
    masked_loss = torch.where(mask, ce_loss, torch.zeros_like(ce_loss))

    # Sum over all valid pairs and normalize
    output = torch.sum(masked_loss, dim=(-2, -1))  # Sum over atom pairs
    num_pairs = n_atoms * (n_atoms - 1)  # Exclude diagonal pairs

    return output / (num_pairs + eps)
