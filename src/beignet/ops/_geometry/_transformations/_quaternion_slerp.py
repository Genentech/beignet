import torch
import torch.testing
from torch import Tensor


def quaternion_slerp(
    input: Tensor,
    time: Tensor,
    rotation: Tensor,
) -> Tensor:
    r"""
    Interpolate between two or more points on a sphere.

    Unlike linear interpolation, which can result in changes in speed when
    interpolating between orientations or positions on a sphere, spherical
    linear interpolation ensures that the interpolation occurs at a constant
    rate and follows the shortest path on the surface of the sphere.
    The process is useful for rotations and orientation interpolation in
    three-dimensional spaces, smoothly transitioning between different
    orientations.

    Mathematically, spherical linear interpolation interpolates between two
    points on a sphere using a parameter $t$, where $t = 0$ represents the
    start point and $t = n$ represents the end point. For two rotation
    quaternions $q_{1}$ and $q_{2}$ representing the start and end
    orientations:

    $$\text{slerp}(q_{1}, q_{2}; t) = q_{1}\frac{\sin((1 - t)\theta)}{\sin(\theta)} + q_{2}\frac{\sin(t\theta)}{\sin(\theta)}$$

    where $\theta$ is the angle between $q_{1}$ and $q_{2}$, and is computed
    using the dot product of $q_{1}$ and $q_{2}$. This formula ensures that the
    interpolation moves along the shortest path on the four-dimensional sphere
    of rotation quaternions, resulting in a smooth and constant-speed rotation.

    Parameters
    ----------
    input : Tensor, shape (..., N)
        Times.

    time : Tensor, shape (..., N)
        Times of the known rotations. At least 2 times must be specified.

    rotation : Tensor, shape (..., N, 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.
    """  # noqa: E501
    if time.shape[-1] != rotation.shape[-2]:
        raise ValueError

    output = torch.empty(
        [*input.shape, 4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for index, t in enumerate(input):
        b = torch.min(torch.nonzero(torch.greater_equal(time, t)))

        if b > 0:
            a = b - 1
        else:
            a = b

        if time[b] == t or b == a:
            output[index] = rotation[b]

            continue

        p, q = time[a], time[b]

        r = (t - p) / (q - p)

        t = rotation[a]
        u = rotation[b]

        v = torch.dot(t, u)

        if v < 0.0:
            u = -u
            v = -v

        if v > 0.9995:
            z = (1.0 - r) * t + r * u
        else:
            x = torch.sqrt(1.0 - v**2.0)

            y = torch.atan2(x, v)

            z = t * torch.sin((1.0 - r) * y) / x + u * torch.sin(r * y) / x

        output[index] = z / torch.linalg.norm(z)

    return output
