import torch
from torch import Tensor


def dihedral_angle(input: Tensor) -> Tensor:
    """Compute dihedral angle from a set of four points.

    Parameters
    ----------
    input: Tensor
        Tensor with shape (*, 4, 3) representing (batches of) four points (p1, p2, p3, p4)

    Returns
    -------
    dihedral: Tensor
        Tensor with shape (*,) representing dihedral angle between
        the planes defined by (p1, p2, p3), (p2, p3, p4).
    """

    if not input.ndim >= 2:
        raise ValueError(f"{input.ndim=} < 2")

    if not input.shape[-2] == 4:
        raise ValueError(f"dihedral requires input.shape[-2] == 4 but {input.shape=}")

    if not input.shape[-1] == 3:
        raise ValueError(f"dihedral requires input.shape[-1] == 3 but {input.shape=}")

    # difference vectors: [a = p2 - p1, b = p3 - p2, c = p4 - p3]
    delta = input[..., 1:, :] - input[..., :-1, :]
    a, b, c = torch.unbind(delta, dim=-2)

    # torsion angle is angle from axb to bxc counterclockwise around b
    axb = torch.cross(a, b, dim=-1)
    bxc = torch.cross(b, c, dim=-1)

    x = torch.sum(bxc * axb, dim=-1)
    y = torch.linalg.vector_norm(b, dim=-1) * torch.sum(bxc * a, dim=-1)
    phi = torch.atan2(y, x)

    return phi
