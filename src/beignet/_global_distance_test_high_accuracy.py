import torch
from torch import Tensor


def global_distance_test_high_accuracy(
    input: Tensor,
    reference: Tensor,
    mask: Tensor | None = None,
    cutoffs: list[float] | None = None,
) -> Tensor:
    r"""
    Compute Global Distance Test High Accuracy (GDT-HA) for protein structure comparison.

    GDT-HA measures structural similarity with higher precision by calculating the average
    fraction of residues that can be superimposed within distance cutoffs of 0.5, 1, 2,
    and 4 Angstroms.

    Parameters
    ----------
    input : Tensor, shape=(..., N, 3)
        Coordinates of the structure to evaluate.
    reference : Tensor, shape=(..., N, 3)
        Coordinates of the reference structure.
    mask : Tensor, shape=(..., N), optional
        Boolean mask indicating which residues to include in the calculation.
        If None, all residues are included.
    cutoffs : list[float], optional
        Distance cutoffs in Angstroms. Default is [0.5, 1.0, 2.0, 4.0].

    Returns
    -------
    Tensor, shape=(...)
        GDT-HA score ranging from 0 to 1, where 1 indicates perfect alignment.

    Examples
    --------
    >>> input = torch.randn(100, 3)
    >>> reference = torch.randn(100, 3)
    >>> gdt_ha = beignet.global_distance_test_high_accuracy(input, reference)
    >>> gdt_ha.shape
    torch.Size([])
    >>> 0 <= gdt_ha <= 1
    True
    """
    if cutoffs is None:
        cutoffs = [0.5, 1.0, 2.0, 4.0]

    # Ensure inputs have the same shape
    if input.shape != reference.shape:
        raise ValueError(
            f"Input and reference must have the same shape, "
            f"got {input.shape} and {reference.shape}"
        )

    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(input.shape[:-1], dtype=torch.bool, device=input.device)

    # Ensure mask has the right shape
    if mask.shape != input.shape[:-1]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match input shape {input.shape[:-1]}"
        )

    # Calculate distances between corresponding residues
    distances = torch.norm(input - reference, dim=-1)  # (..., N)

    # Apply mask
    masked_distances = torch.where(mask, distances, torch.inf)

    # Vectorized computation for better performance
    mask_float = mask.to(torch.get_default_dtype())
    mask_sum = torch.sum(mask_float, dim=-1, keepdim=True)  # (..., 1)

    # Create cutoffs tensor for vectorized operations
    cutoffs_tensor = torch.tensor(cutoffs, dtype=input.dtype, device=input.device)

    # Expand dimensions for broadcasting
    distances_expanded = masked_distances.unsqueeze(-1)  # (..., N, 1)
    cutoffs_expanded = cutoffs_tensor.view(
        *([1] * (distances_expanded.ndim - 1)), -1
    )  # (..., 1, len(cutoffs))

    # Compute all comparisons at once
    within_cutoff = (distances_expanded <= cutoffs_expanded).to(
        torch.get_default_dtype()
    )  # (..., N, len(cutoffs))

    # Apply mask and sum over residues
    numerator = torch.sum(
        within_cutoff * mask_float.unsqueeze(-1), dim=-2
    )  # (..., len(cutoffs))

    # Avoid division by zero
    scores = torch.where(
        mask_sum > 0, numerator / mask_sum, torch.zeros_like(numerator)
    )

    # Calculate GDT-HA as average over all cutoffs
    gdt_ha = torch.mean(scores, dim=-1)

    return gdt_ha
