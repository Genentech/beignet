import re

import torch
from torch import Tensor


def euler_angle_to_rotation_vector(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    r"""
    Convert Euler angles to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Euler angles.

    axes : str
        Axes. One to three characters belonging to the set :math:`\{X, Y, Z\}`
        for intrinsic rotations, or :math:`\{x, y, z\}` for extrinsic
        rotations. Extrinsic and intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees and returned
        rotation vector magnitudes are in degrees. Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Rotation vectors.
    """
    num_axes = len(axes)

    if num_axes < 1 or num_axes > 3:
        raise ValueError

    intrinsic = re.match(r"^[XYZ]{1,3}$", axes) is not None
    extrinsic = re.match(r"^[xyz]{1,3}$", axes) is not None

    if not (intrinsic or extrinsic):
        raise ValueError

    if any(axes[i] == axes[i + 1] for i in range(num_axes - 1)):
        raise ValueError

    if degrees:
        input = torch.deg2rad(input)

    if len(axes.lower()) == 1:
        match input.ndim:
            case 0:
                input = torch.reshape(input, [1, 1])
            case 1:
                input = input[:, None]
            case 2 if input.shape[-1] != 1:
                raise ValueError
            case _:
                raise ValueError

    else:
        if input.ndim not in [1, 2] or input.shape[-1] != len(axes.lower()):
            raise ValueError

        if input.ndim == 1:
            input = input[None, :]

    if input.ndim != 2 or input.shape[-1] != len(axes.lower()):
        raise ValueError

    x = torch.zeros(
        [input.shape[0], 4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    match axes.lower()[0]:
        case "x":
            m = 0
        case "y":
            m = 1
        case "z":
            m = 2
        case _:
            raise ValueError

    for j in range(input[:, 0].shape[0]):
        x[j, 3] = torch.cos(input[:, 0][j] / 2)
        x[j, m] = torch.sin(input[:, 0][j] / 2)

    for j in range(1, len(axes)):
        y = torch.zeros(
            [input.shape[0], 4],
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )

        z = torch.empty(
            [max(y.shape[0], x.shape[0]), 4],
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )

        match axes.lower()[j]:
            case "x":
                m = 0
            case "y":
                m = 1
            case "z":
                m = 2
            case _:
                raise ValueError

        for k in range(input[:, j].shape[0]):
            y[k, 3] = torch.cos(input[:, j][k] / 2)
            y[k, m] = torch.sin(input[:, j][k] / 2)

        if intrinsic:
            if x.shape[0] == 1:
                for k in range(max(x.shape[0], y.shape[0])):
                    q = y[k, 1]
                    r = y[k, 2]
                    s = y[k, 3]
                    p = y[k, 0]

                    t = x[0, 0]
                    u = x[0, 1]
                    v = x[0, 2]
                    w = x[0, 3]

                    z[k, 0] = w * p + s * t + u * r - v * q
                    z[k, 1] = w * q + s * u + v * p - t * r
                    z[k, 2] = w * r + s * v + t * q - u * p
                    z[k, 3] = w * s - t * p - u * q - v * r
            elif y.shape[0] == 1:
                for k in range(max(x.shape[0], y.shape[0])):
                    p = y[0, 0]
                    q = y[0, 1]
                    r = y[0, 2]
                    s = y[0, 3]

                    t = x[k, 0]
                    u = x[k, 1]
                    v = x[k, 2]
                    w = x[k, 3]

                    z[k, 0] = w * p + s * t + u * r - v * q
                    z[k, 1] = w * q + s * u + v * p - t * r
                    z[k, 2] = w * r + s * v + t * q - u * p
                    z[k, 3] = w * s - t * p - u * q - v * r
            else:
                for k in range(max(x.shape[0], y.shape[0])):
                    p = y[k, 0]
                    q = y[k, 1]
                    r = y[k, 2]
                    s = y[k, 3]

                    t = x[k, 0]
                    u = x[k, 1]
                    v = x[k, 2]
                    w = x[k, 3]

                    z[k, 0] = w * p + s * t + u * r - v * q
                    z[k, 1] = w * q + s * u + v * p - t * r
                    z[k, 2] = w * r + s * v + t * q - u * p
                    z[k, 3] = w * s - t * p - u * q - v * r

            x = z
        else:
            if y.shape[0] == 1:
                for k in range(max(y.shape[0], x.shape[0])):
                    p = y[0, 0]
                    q = y[0, 1]
                    r = y[0, 2]
                    s = y[0, 3]

                    t = x[k, 0]
                    u = x[k, 1]
                    v = x[k, 2]
                    w = x[k, 3]

                    z[k, 0] = s * t + w * p + q * v - r * u
                    z[k, 1] = s * u + w * q + r * t - p * v
                    z[k, 2] = s * v + w * r + p * u - q * t
                    z[k, 3] = s * w - p * t - q * u - r * v
            elif x.shape[0] == 1:
                for k in range(max(y.shape[0], x.shape[0])):
                    t = x[0, 0]
                    u = x[0, 1]
                    v = x[0, 2]
                    w = x[0, 3]

                    p = y[k, 0]
                    q = y[k, 1]
                    r = y[k, 2]
                    s = y[k, 3]

                    z[k, 0] = s * t + w * p + q * v - r * u
                    z[k, 1] = s * u + w * q + r * t - p * v
                    z[k, 2] = s * v + w * r + p * u - q * t
                    z[k, 3] = s * w - p * t - q * u - r * v
            else:
                for k in range(max(y.shape[0], x.shape[0])):
                    p = y[k, 0]
                    q = y[k, 1]
                    r = y[k, 2]
                    s = y[k, 3]

                    t = x[k, 0]
                    u = x[k, 1]
                    v = x[k, 2]
                    w = x[k, 3]

                    z[k, 0] = s * t + w * p + q * v - r * u
                    z[k, 1] = s * u + w * q + r * t - p * v
                    z[k, 2] = s * v + w * r + p * u - q * t
                    z[k, 3] = s * w - p * t - q * u - r * v

            x = z

    output = torch.empty(
        [x.shape[0], 3],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(x.shape[0]):
        a = x[j, 0]
        b = x[j, 1]
        c = x[j, 2]
        d = x[j, 3]

        if d == 0 and (a == 0 and (b == 0 and c < 0 or b < 0) or a < 0) or d < 0:
            x[j] = -x[j]

        y = 2.0 * torch.atan2(torch.sqrt(a**2.0 + b**2.0 + c**2.0), d**1.0)

        if y < 0.001:
            y = 2.0 + y**2.0 / 12.0 + 7.0 * y**2.0 * y**2.0 / 2880.0
        else:
            y = y / torch.sin(y / 2.0)

        output[j] = x[j, :-1] * y

    if degrees:
        output = torch.rad2deg(output)

    return output
