import torch
import torch._dynamo
import torch.nn.functional as F
from torch import Tensor


def distogram_loss(
    logits: Tensor,
    target_distances: Tensor,
    mask: Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    n_bins: int = 64,
    reduction: str = "mean",
) -> Tensor:
    r"""
    Compute the distogram loss for protein structure prediction.

    The distogram loss is a cross-entropy loss between predicted distance
    distributions and target distances. It discretizes continuous distances
    into bins and treats the problem as a classification task.

    Parameters
    ----------
    logits : Tensor, shape=(..., N, N, n_bins)
        Predicted logits for distance bins. N is the number of residues.
    target_distances : Tensor, shape=(..., N, N)
        True distances between residue pairs.
    mask : Tensor, shape=(..., N, N)
        Binary mask indicating valid residue pairs (1 for valid, 0 for invalid).
    min_bin : float, default=2.3125
        Minimum distance for binning (in Angstroms).
    max_bin : float, default=21.6875
        Maximum distance for binning (in Angstroms).
    n_bins : int, default=64
        Number of distance bins.
    reduction : str, default="mean"
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of valid pairs,
        'sum': the output will be summed.

    Returns
    -------
    loss : Tensor
        The distogram loss. Shape depends on reduction:
        - 'none': shape=(..., N, N)
        - 'mean' or 'sum': scalar

    Examples
    --------
    >>> batch_size, n_residues, n_bins = 2, 50, 64
    >>> logits = torch.randn(batch_size, n_residues, n_residues, n_bins)
    >>> distances = torch.rand(batch_size, n_residues, n_residues) * 20 + 2
    >>> mask = torch.ones(batch_size, n_residues, n_residues)
    >>> loss = distogram_loss(logits, distances, mask)
    >>> loss.shape
    torch.Size([])
    """
    # Validate inputs
    if min_bin >= max_bin:
        raise ValueError(
            f"min_bin must be less than max_bin, got {min_bin} >= {max_bin}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError(f"Invalid reduction: {reduction}")

    # Check shape compatibility
    if logits.shape[:-1] != target_distances.shape:
        raise ValueError(
            f"Shape mismatch: logits shape {logits.shape} incompatible with "
            f"target_distances shape {target_distances.shape}"
        )
    if logits.shape[:-1] != mask.shape:
        raise ValueError(
            f"Shape mismatch: logits shape {logits.shape} incompatible with "
            f"mask shape {mask.shape}"
        )

    # Optimized bin computation
    # Instead of creating full bin edges, compute bin indices directly
    # This avoids the searchsorted operation which can be slow
    bin_width = (max_bin - min_bin) / n_bins

    # Compute bin indices directly - much faster than searchsorted
    # Clamp first to avoid edge cases, then compute indices
    clamped_distances = torch.clamp(target_distances, min_bin, max_bin - 1e-6)
    target_bins = ((clamped_distances - min_bin) / bin_width).long()

    # Early exit for empty mask (removed for torch.compile compatibility)

    # Check if we're running under torch.compile
    # The sparse path is faster but not compatible with torch.compile
    is_compiling = torch._dynamo.is_compiling()

    # Use sparse operations when mask has many zeros (common case)
    mask_bool = mask.bool()
    sparsity = 1.0 - mask_bool.float().mean().item()

    if not is_compiling and sparsity > 0.5 and reduction != "none":
        # Sparse path: only compute loss for valid pairs
        # This is much faster when many pairs are masked
        mask_flat_bool = mask_bool.reshape(-1)
        valid_indices = mask_flat_bool.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

        logits_valid = logits.reshape(-1, n_bins)[valid_indices]
        targets_valid = target_bins.reshape(-1)[valid_indices]

        loss_valid = F.cross_entropy(logits_valid, targets_valid, reduction="none")

        if reduction == "sum":
            return loss_valid.sum()
        else:  # mean
            return loss_valid.mean()

    # Dense path: compute for all pairs then mask
    # Reshape for cross_entropy: (batch * N * N, n_bins)
    logits_flat = logits.reshape(-1, n_bins)
    target_bins_flat = target_bins.reshape(-1)

    # Compute unreduced loss
    loss_flat = F.cross_entropy(logits_flat, target_bins_flat, reduction="none")

    # Apply mask efficiently
    if mask.dtype != loss_flat.dtype:
        mask_flat = mask.reshape(-1).to(loss_flat.dtype)
    else:
        mask_flat = mask.reshape(-1)

    loss_flat = loss_flat * mask_flat

    # Reshape back to original shape for "none" reduction
    if reduction == "none":
        return loss_flat.reshape(logits.shape[:-1])

    # Apply reduction for dense path
    elif reduction == "sum":
        return loss_flat.sum()
    else:  # mean
        # Mean over valid pairs only
        n_valid = mask_flat.sum()
        # Avoid division by zero with torch.where for torch.compile compatibility
        return torch.where(
            n_valid > 0,
            loss_flat.sum() / n_valid,
            torch.tensor(0.0, dtype=loss_flat.dtype, device=loss_flat.device),
        )
