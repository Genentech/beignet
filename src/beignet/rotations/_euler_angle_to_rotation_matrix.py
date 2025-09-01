import torch
from torch import Tensor


def euler_angle_to_rotation_matrix(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    r"""
    Convert Euler angles to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Euler angles.

    axes : str
        Axes. One to three characters belonging to the set :math:`\{X, Y, Z\}`
        for intrinsic rotations, or :math:`\{x, y, z\}` for extrinsic
        rotations. Extrinsic and intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3, 3)
        Rotation matrices.
    """
    if degrees:
        input = torch.deg2rad(input)

    output = torch.empty(
        [input.shape[0], 3, 3],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j, axis in enumerate(axes):
        a = torch.cos(input[..., j])
        b = torch.sin(input[..., j])

        p = torch.full_like(a, 1.0)
        q = torch.full_like(a, 0.0)

        match axis.lower():
            case "x":
                x = [
                    torch.stack([+p, +q, +q], dim=-1),
                    torch.stack([+q, +a, -b], dim=-1),
                    torch.stack([+q, +b, +a], dim=-1),
                ]
            case "y":
                x = [
                    torch.stack([+a, +q, +b], dim=-1),
                    torch.stack([+q, +p, +q], dim=-1),
                    torch.stack([-b, +q, +a], dim=-1),
                ]
            case "z":
                x = [
                    torch.stack([+a, -b, +q], dim=-1),
                    torch.stack([+b, +a, +q], dim=-1),
                    torch.stack([+q, +q, +p], dim=-1),
                ]
            case _:
                raise ValueError

        x = torch.stack(x, dim=-2)

        if j == 0:
            output = x
        else:
            if axes.islower():
                output = x @ output
            else:
                output = output @ x

    return output
