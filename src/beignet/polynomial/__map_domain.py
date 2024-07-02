from torch import Tensor


def _map_domain(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    (a, b), (c, d) = y, z

    return (b * c - a * d) / (b - a) + (d - c) / (b - a) * x
