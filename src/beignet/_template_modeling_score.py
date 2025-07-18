import torch
from torch import Tensor


def template_modeling_score(
    input: Tensor,
    target: Tensor,
    weights: Tensor | None = None,
    d0: float | None = None,
    aligned: bool = False,
) -> Tensor:
    r"""
    Calculate the Template Modeling score (TM-score) between two protein structures.

    The TM-score is a metric for measuring the structural similarity between two
    protein structures. It has the value in (0,1], where 1 indicates a perfect
    match between two structures. TM-score is more sensitive to the global topology
    than local structural variations and is length-independent for random structure
    pairs.

    Parameters
    ----------
    input : Tensor, shape (..., N, 3)
        Coordinates of the first protein structure, typically C-alpha atoms.
        N is the number of residues.
    target : Tensor, shape (..., N, 3)
        Coordinates of the second protein structure to compare against.
        Must have the same number of residues as input.
    weights : Tensor, shape (..., N), optional
        Weights for each residue. If provided, residues with zero weight are
        excluded from the calculation. Default is None (all residues included).
    d0 : float, optional
        Normalization factor. If None, it is calculated as:
        d0 = 1.24 * (N - 15)^(1/3) - 1.8, where N is the number of residues.
        This is the standard formula for TM-score normalization.
    aligned : bool, optional
        If True, assumes the structures are already optimally aligned.
        If False (default), performs optimal superposition using Kabsch algorithm.

    Returns
    -------
    score : Tensor, shape (...)
        The TM-score between the two structures. Values range from 0 to 1,
        where 1 indicates identical structures.

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> # Two protein structures with 100 residues each
    >>> structure1 = torch.randn(100, 3)
    >>> structure2 = structure1 + 0.5 * torch.randn(100, 3)  # Add some noise
    >>> score = beignet.template_modeling_score(structure1, structure2)
    >>> assert 0 < score < 1

    >>> # With batch dimension
    >>> batch_size = 10
    >>> structures1 = torch.randn(batch_size, 100, 3)
    >>> structures2 = structures1 + 0.5 * torch.randn(batch_size, 100, 3)
    >>> scores = beignet.template_modeling_score(structures1, structures2)
    >>> assert scores.shape == (batch_size,)

    Notes
    -----
    The TM-score is defined as:

    TM-score = 1/N * sum_i 1/(1 + (d_i/d0)^2)

    where d_i is the distance between the i-th pair of residues after optimal
    superposition, N is the number of residues, and d0 is a normalization factor
    that depends on the protein length.

    References
    ----------
    Zhang, Y. and Skolnick, J. (2004). Scoring function for automated assessment
    of protein structure template quality. Proteins, 57: 702-710.
    """
    # Validate inputs
    if input.shape != target.shape:
        raise ValueError(
            f"Input and target must have the same shape, got {input.shape} and {target.shape}"
        )

    if input.shape[-1] != 3:
        raise ValueError(
            f"Last dimension must be 3 (x, y, z coordinates), got {input.shape[-1]}"
        )

    *batch_dims, n_residues, _ = input.shape

    # Handle weights
    if weights is None:
        weights = torch.ones(
            *batch_dims, n_residues, dtype=input.dtype, device=input.device
        )
    else:
        if weights.shape != (*batch_dims, n_residues):
            raise ValueError(
                f"Weights shape {weights.shape} doesn't match expected shape {(*batch_dims, n_residues)}"
            )
        weights = weights.to(dtype=input.dtype)

    # Calculate effective number of residues
    n_eff = weights.sum(dim=-1, keepdim=True)

    # Calculate d0 if not provided
    if d0 is None:
        # Standard TM-score d0 formula
        d0 = 1.24 * torch.pow(torch.clamp(n_eff - 15, min=1), 1.0 / 3.0) - 1.8
        d0 = torch.clamp(d0, min=0.5)  # Ensure d0 is at least 0.5
    else:
        # Convert d0 to tensor with proper shape
        d0_value = torch.as_tensor(d0, dtype=input.dtype, device=input.device)
        if batch_dims:
            d0 = d0_value.expand(*batch_dims, 1)
        else:
            d0 = d0_value.view(1)

    # Perform alignment if needed
    if not aligned:
        from ._kabsch import kabsch

        # Mask out zero-weight residues for alignment
        mask = weights > 0
        masked_weights = weights * mask

        # Compute optimal alignment
        t, r = kabsch(input, target, weights=masked_weights, keepdim=False)
        # Add batch dimension back if needed for matmul
        if r.dim() == 2:  # Single structure case
            aligned_target = target @ r.T + t
        else:  # Batch case
            # Use einsum for better performance with batches
            aligned_target = torch.einsum(
                "...ij,...jk->...ik", target, r.transpose(-2, -1)
            )
            aligned_target += t.unsqueeze(-2)
    else:
        aligned_target = target

    # Calculate distances - optimized version
    # Use squared distances to avoid sqrt computation
    diff = input - aligned_target
    distances_squared = (diff * diff).sum(dim=-1)  # Shape: (..., N)

    # Calculate TM-score - optimized version
    # TM-score = 1/N * sum_i 1/(1 + (d_i/d0)^2)
    # Rewrite as: d0²/(d0² + d²) to avoid division
    d0_squared = d0 * d0
    d0_squared = (
        d0_squared.unsqueeze(-1)
        if d0_squared.dim() < distances_squared.dim()
        else d0_squared
    )
    score_per_residue = d0_squared / (d0_squared + distances_squared)

    # Apply weights and normalize
    weighted_sum = (score_per_residue * weights).sum(dim=-1)
    tm_score_value = weighted_sum / n_eff.squeeze(-1)

    # Handle case where all weights are zero (avoid NaN)
    tm_score_value = torch.where(
        n_eff.squeeze(-1) == 0, torch.zeros_like(tm_score_value), tm_score_value
    )

    return tm_score_value
