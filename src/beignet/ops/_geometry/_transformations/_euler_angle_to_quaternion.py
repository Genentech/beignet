import re

import torch
from torch import Tensor


def euler_angle_to_quaternion(
    input: Tensor,
    axes: str,
    degrees: bool = False,
    canonical: bool | None = False,
) -> Tensor:
    r"""
    Convert Euler angles to rotation quaternions.

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

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    output : Tensor, shape=(..., 4)
        Rotation quaternions.
    """
    intrinsic = re.match(r"^[XYZ]{1,3}$", axes) is not None

    if degrees:
        input = torch.deg2rad(input)

    if len(axes) == 1:
        if input.ndim == 0:
            input = input.reshape([1, 1])
        elif input.ndim == 1:
            input = input[:, None]
        elif input.ndim == 2 and input.shape[-1] != 1:
            raise ValueError
        elif input.ndim > 2:
            raise ValueError
    else:
        if input.ndim not in [1, 2] or input.shape[-1] != len(axes):
            raise ValueError

        if input.ndim == 1:
            input = input[None, :]

    if input.ndim != 2 or input.shape[-1] != len(axes):
        raise ValueError

    output = torch.zeros(
        [input.shape[0], 4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    match axes.lower()[0]:
        case "x":
            k = 0
        case "y":
            k = 1
        case "z":
            k = 2
        case _:
            raise ValueError

    for j in range(input[:, 0].shape[0]):
        output[j, 3] = torch.cos(input[:, 0][j] / 2)
        output[j, k] = torch.sin(input[:, 0][j] / 2)

    z = output

    c = torch.empty(
        [3],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(1, len(axes.lower())):
        y = torch.zeros(
            [input.shape[0], 4],
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )

        r = torch.empty(
            [max(y.shape[0], z.shape[0]), 4],
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )

        match axes.lower()[j]:
            case "x":
                p = 0
            case "y":
                p = 1
            case "z":
                p = 2
            case _:
                raise ValueError

        for k in range(input[:, j].shape[0]):
            y[k, 3] = torch.cos(input[:, j][k] / 2)
            y[k, p] = torch.sin(input[:, j][k] / 2)

        if intrinsic:
            for k in range(max(y.shape[0], z.shape[0])):
                c[0] = z[k, 1] * y[k, 2] - z[k, 2] * y[k, 1]
                c[1] = z[k, 2] * y[k, 0] - z[k, 0] * y[k, 2]
                c[2] = z[k, 0] * y[k, 1] - z[k, 1] * y[k, 0]

                t = z[k, 0]
                u = z[k, 1]
                v = z[k, 2]
                w = z[k, 3]

                r[k, 0] = w * y[k, 0] + y[k, 3] * t + c[0]
                r[k, 1] = w * y[k, 1] + y[k, 3] * u + c[1]
                r[k, 2] = w * y[k, 2] + y[k, 3] * v + c[2]
                r[k, 3] = w * y[k, 3] - t * y[k, 0] - u * y[k, 1] - v * y[k, 2]

            z = r
        else:
            for k in range(max(y.shape[0], z.shape[0])):
                c[0] = y[k, 1] * z[k, 2] - y[k, 2] * z[k, 1]
                c[1] = y[k, 2] * z[k, 0] - y[k, 0] * z[k, 2]
                c[2] = y[k, 0] * z[k, 1] - y[k, 1] * z[k, 0]

                t = z[k, 0]
                u = z[k, 1]
                v = z[k, 2]
                w = z[k, 3]

                r[k, 0] = y[k, 3] * t + w * y[k, 0] + c[0]
                r[k, 1] = y[k, 3] * u + w * y[k, 1] + c[1]
                r[k, 2] = y[k, 3] * v + w * y[k, 2] + c[2]
                r[k, 3] = y[k, 3] * w - y[k, 0] * t - y[k, 1] * u - y[k, 2] * v

            z = r

    if canonical:
        for j in range(z.shape[0]):
            a = z[j, 0]
            b = z[j, 1]
            c = z[j, 2]
            d = z[j, 3]

            if d == 0 and (a == 0 & (b == 0 & c < 0 | b < 0) | a < 0) | d < 0:
                z[j] = -z[j]

    return z
