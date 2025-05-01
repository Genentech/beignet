import torch
from torch import Tensor


def _weighted_mean(input: Tensor, weights: Tensor, dim: int = 0, keepdim: bool = False):
    a = torch.sum(input * weights, dim=dim, keepdim=keepdim)
    b = torch.sum(weights, dim=dim, keepdim=keepdim)
    out = torch.where(b == 0.0, 0.0, a / b)  # avoid nan in input to svd
    return out


def kabsch(
    x: Tensor,
    y: Tensor,
    *,
    weights: Tensor | None = None,
    driver: str | None = None,
    keepdim: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute an optimal rotation and translation between two paired sets of points.

    Given tensors `x` and `y` find the rigid transformation `T = (t, r)`
    which minimizes the RMSD between x and T(y).

    Parameters
    ----------
    x : Tensor
        Shape (*, N, D)
    y : Tensor
        Shape (*, N, D)
    weights : Tensor | None
        Shape (*, N)

    where `*` is any number of batch dimension, `N` is the number of points in each batch
    and `D` is the number of "spatial" dimensions.

    Returns
    -------
    t: Tensor
        Optimal translation. Shape (*, 1, 3)
    r: Tensor
        Optimal rotation matrix. Shape (*, 1, 3, 3)
    """
    _, D = x.shape[-2:]
    assert y.shape == x.shape

    if weights is None:
        weights = torch.ones(*x.shape, dtype=x.dtype, device=x.device)
    else:
        weights = weights.unsqueeze(-1).to(dtype=x.dtype)

    x_mu = _weighted_mean(x, weights, dim=-2, keepdim=True)
    y_mu = _weighted_mean(y, weights, dim=-2, keepdim=True)

    x_c = x - x_mu
    y_c = y - y_mu

    H = torch.einsum("...mi,...mj,...mj->...ij", y_c, x_c, weights)
    u, _, vh = torch.linalg.svd(H, driver=driver)

    r = torch.einsum("...ki,...jk->...ij", vh, u)  # V U^T

    # remove reflections
    # we adjust the last row because this corresponds to smallest singular value
    sign = torch.cat(
        [
            torch.ones(*x.shape[:-2], D - 1, device=x.device, dtype=x.dtype),
            torch.linalg.det(r).unsqueeze(-1),
        ],
        dim=-1,
    )
    r = torch.einsum("...ki,...k,...jk->...ij", vh, sign, u).unsqueeze(-3)  # V S U^T
    t = x_mu - torch.einsum("...ij,...j->...i", r, y_mu)

    if not keepdim:
        r = torch.squeeze(r, -3)
        t = torch.squeeze(t, -2)

    return t, r
