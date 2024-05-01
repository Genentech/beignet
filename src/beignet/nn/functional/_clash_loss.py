import torch
from torch import Tensor


def clash_loss(
    input: Tensor,
    target: (Tensor, Tensor),
    mask: Tensor,
    tighten=0.0,
    epsilon=1e-10,
) -> (Tensor, Tensor, Tensor):
    r"""
    A one-sided flat-bottom-potential, that penalizes steric clashes:

    $$\mathcal{L}_{\text{clash}}=\sum_{i=1}^{N_{\text{non-bonded}}}\max{
        \left(\text{distance }_{\text{Van der Waals radii}}^{i}-
        \tau-
        \text{distance }_{\text{predicted}}^{i},0\right)},$$

    where $N_{\text{non-bonded pairs}}$ is the number of all non-bonded atom
    pairs, $\text{distance }_{\text{predicted}}^{i}$ is the distance of two
    non-bonded atoms in the predicted structure, and
    $\text{distance }_{\text{Van der Waals radii}}^{i}$ is the “clashing
    distance” of two non-bonded atoms according to their Van der Waals radii.
    The tolerance, $\tau$, $1.5\text{\r{A}}$.

    Parameters
    ----------
    input : Tensor, shape=(..., N, 14, 3)
        Predicted positions of atoms in global prediction frame.

    target : Tensor, shape=(..., N, 14), Tensor, shape=(..., N, 14)
        Lower and upper bound on allowed distances.

    mask : Tensor, shape=(..., N, 14)
        Mask denoting whether atom at positions exists for given amino acid type.

    tighten : float, optional
        Extra factor to tighten loss. Default, 0.0.

    epsilon : float, optional
        Small value to avoid division by zero. Default, 1e-10.

    Returns
    -------
    output : Tensor, shape=(..., N, 14)
        Sum of all clash losses per atom.

    mask : Tensor, shape=(..., N, 14)
        Whether atom clashes with any other atom.

    clashes : Tensor, shape=(..., N)
        Number of clashes per atom.
    """
    distance_mask = torch.eye(14)
    distance_mask = distance_mask[None]
    distance_mask = 1.0 - distance_mask
    shape = [*((1,) * len(mask.shape[:-2])), *distance_mask.shape]
    distance_mask = torch.reshape(distance_mask, shape)
    distance_mask = distance_mask * mask[..., :, :, None]
    distance_mask = distance_mask * mask[..., :, None, :]

    distance = input[..., :, :, None, :] - input[..., :, None, :, :]
    distance = torch.sqrt(torch.sum(distance**2, dim=-1) + epsilon)

    a, b = target

    a = torch.nn.functional.relu((a + tighten) - distance)
    b = torch.nn.functional.relu(distance - (b - tighten))

    loss = (a + b) * distance_mask

    violations = ((distance < a) | (distance > b)) * distance_mask

    return (
        torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1),
        torch.maximum(
            torch.max(violations, dim=-2)[0],
            torch.max(violations, dim=-1)[0],
        ),
        torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1),
    )
