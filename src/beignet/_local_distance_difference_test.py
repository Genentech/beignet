import torch
from torch import Tensor


def local_distance_difference_test(
    input: Tensor,
    target: Tensor,
    mask: Tensor,
    cutoff: float | Tensor = 15.0,
):
    """
    The Local Distance Difference Test (LDDT) is a superposition-free score
    used to evaluate the local quality of protein structures by comparing the
    distances between atoms in a model to those in a reference structure,
    without requiring global superposition.

    Parameters
    ----------

    input : Tensor
        The input tensor representing the predicted structure.
        Shape: (..., N, 3) where N is the number of atoms.

    target : Tensor
        The target tensor representing the reference structure.
        Shape: (..., N, 3) where N is the number of atoms.

    mask : Tensor
        A binary mask indicating the presence of atoms in the input structure.
        Shape: (..., N) where N is the number of atoms.

    cutoff : float | Tensor, optional
        The cutoff distance for considering atoms in the calculation.
        Default is 15.0.

    Returns
    -------
    Tensor
        The Local Distance Difference Test (LDDT) score.
    """
    epsilon = torch.finfo(input.dtype).eps

    n = mask.shape[-2]

    a = input[..., None, :] - input[..., None, :, :]
    a = a**2
    a = torch.sum(a, dim=-1)
    a = a + epsilon
    a = torch.sqrt(a)

    target = target[..., None, :] - target[..., None, :, :]
    target = target**2
    target = torch.sum(target, dim=-1) + epsilon
    target = torch.sqrt(target)

    x = target < cutoff
    x = x * mask
    indicies = [1, 0]

    dimensions = [*range(len(mask.shape[: -1 * len(indicies)]))]

    for index in indicies:
        dimensions.append(-1 * len(indicies) + index)

    permuted_mask = torch.permute(mask, dimensions)

    x = x * permuted_mask
    z = torch.eye(n, device=mask.device)
    z = 1.0 - z
    x = x * z

    dist_l1 = torch.abs(target - a)

    m = (dist_l1 < 0.5).to(dtype=dist_l1.dtype)
    n = (dist_l1 < 1.0).to(dtype=dist_l1.dtype)
    o = (dist_l1 < 2.0).to(dtype=dist_l1.dtype)
    p = (dist_l1 < 4.0).to(dtype=dist_l1.dtype)

    score = m + n
    score = score + o
    score = score + p
    score = score * 0.25

    per_residue = False

    if per_residue:
        dim = [-1]
    else:
        dim = [-2, -1]

    y = torch.sum(x, dim=dim)
    y = y + epsilon

    norm = 1.0 / y

    output = x * score
    output = torch.sum(output, dim=dim)
    output = output + epsilon
    output = output * norm

    return output
