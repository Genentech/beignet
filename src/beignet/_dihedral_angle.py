import torch
from torch import Tensor


def dihedral_angle(input: Tensor) -> Tensor:
    """Compute dihedral angle from a set of four points.

    Given an input tensor with shape (*, 4, 3) representing points (p1, p2, p3, p4)
    compute the dihedral angle between the planes defined by (p1, p2, p3), (p2, p3, p4).

    Parameters
    ----------
    input: Tensor
        Shape (*, 4, 3)

    Returns
    -------
    torsion: Tensor
        Shape (*,)
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

    # orthogonal basis in plane perpendicular to b
    # NOTE v1 and v2 are not unit but have the same magnitude
    v1 = axb
    v2 = torch.cross(torch.nn.functional.normalize(b, dim=-1), axb, dim=-1)

    x = torch.sum(bxc * v1, dim=-1)
    y = torch.sum(bxc * v2, dim=-1)
    phi = torch.atan2(y, x)

    return phi
