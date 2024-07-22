import math
import re

import torch
from torch import Tensor


def quaternion_to_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    r"""
    Convert rotation quaternions to Euler angles.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Euler angles. The returned Euler angles are in the range:

            * First angle: :math:`(-180, 180]` degrees (inclusive)
            * Second angle:
                * :math:`[-90, 90]` degrees if all axes are different
                  (e.g., :math:`xyz`)
                * :math:`[0, 180]` degrees if first and third axes are the same
                  (e.g., :math:`zxz`)
            * Third angle: :math:`[-180, 180]` degrees (inclusive)
    """
    epsilon = torch.finfo(input.dtype).eps

    output = torch.empty(
        [input.shape[0], 3],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    extrinsic = re.match(r"^[xyz]{1,3}$", axes) is not None

    axes = axes.lower()

    if not extrinsic:
        axes = axes[::-1]

    match axes[0]:
        case "x":
            p = 0
        case "y":
            p = 1
        case "z":
            p = 2
        case _:
            raise ValueError

    match axes[1]:
        case "x":
            q = 0
        case "y":
            q = 1
        case "z":
            q = 2
        case _:
            raise ValueError

    match axes[2]:
        case "x":
            r = 0
        case "y":
            r = 1
        case "z":
            r = 2
        case _:
            raise ValueError

    if p == r:
        r = 3 - p - q

    s = (p - q) * (q - r) * (r - p) // 2

    for j in range(input.shape[0]):
        if p == r:
            t = input[j, 3]
            u = input[j, p]
            v = input[j, q]
            w = input[j, r] * s
        else:
            t = input[j, 3] - input[j, q]
            u = input[j, p] + input[j, r] * s
            v = input[j, q] + input[j, 3]
            w = input[j, r] * s - input[j, p]

        if extrinsic:
            a = 0
            c = 2
        else:
            a = 2
            c = 0

        output[j, 1] = 2.0 * torch.atan2(torch.hypot(v, w), torch.hypot(t, u))

        match output[j, 1]:
            case _ if abs(output[j, 1]) < epsilon:
                output[j, 0] = 2.0 * torch.atan2(u, t)
                output[j, 2] = 0.0
            case _ if abs(output[j, 1] - math.pi) < epsilon:
                if extrinsic:
                    output[j, 0] = 2.0 * -torch.atan2(w, v)
                else:
                    output[j, 0] = 2.0 * +torch.atan2(w, v)

                output[j, 2] = 0.0
            case _:
                output[j, a] = torch.atan2(u, t) - torch.atan2(w, v)
                output[j, c] = torch.atan2(u, t) + torch.atan2(w, v)

        if not p == r:
            output[j, 1] = output[j, 1] - math.pi / 2.0
            output[j, c] = output[j, c] * s

        for k in range(3):
            if output[j, k] <= -math.pi:
                output[j, k] = output[j, k] + math.tau

            if output[j, k] >= +math.pi:
                output[j, k] = output[j, k] - math.tau

    if degrees:
        output = torch.rad2deg(output)

    return output
