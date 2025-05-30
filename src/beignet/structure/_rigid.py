import functools
import operator

import optree
import torch
from optree.dataclasses import dataclass
from torch import Tensor

from beignet import identity_matrix, kabsch, quaternion_to_rotation_matrix

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@dataclass(namespace="beignet")
class Rigid:
    t: Tensor  # [...,3]
    r: Tensor  # [...,3,3]

    def __call__(self, x: Tensor) -> Tensor:
        return self.t + torch.einsum("...ij,...j->...i", self.r, x)

    def __getitem__(self, key):
        return optree.tree_map(
            lambda x: operator.getitem(x, key), self, namespace="beignet"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.t.shape[:-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def rand(cls, *size, dtype=None, device=None):
        t = torch.randn((*size, 3), dtype=dtype, device=device)
        q = torch.nn.functional.normalize(
            torch.randn((*size, 4), dtype=dtype, device=device), dim=-1
        )
        r = quaternion_to_rotation_matrix(q)
        return cls(t, r)

    @classmethod
    def identity(cls, *size, dtype=None, device=None):
        t = torch.zeros(*size, 3, dtype=dtype, device=device)
        r = identity_matrix(3, size)
        return cls(t, r)

    @classmethod
    def kabsch(
        cls,
        x: Tensor,
        y: Tensor,
        *,
        weights: Tensor | None = None,
        driver: str | None = None,
        keepdim: bool = True,
    ):
        """Compute the optimal rigid transformation between two paired sets of points.

        Examples
        --------
        >>> B, N, D = (11, 4, 3)
        >>> T = Rigid.rand(B)
        >>> x = torch.randn(B,N,D)
        >>> y = T(x)
        >>> T_kabsch = Rigid.kabsch(y, x)
        >>> torch.testing.assert_close(T_kabsch.t, T.t)
        >>> torch.testing.assert_close(T_kabsch.r, T.r)

        In this example we see that the kabsch algorithm successfully recovers the exact
        translation/rotation describing the transformation between x_ref and y.
        """
        t, r = kabsch(x, y, weights=weights, driver=driver, keepdim=keepdim)
        return cls(t, r)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, Rigid)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def to(self, dtype=None, device=None):
        return optree.tree_map(
            lambda x: x.to(dtype=dtype, device=device), self, namespace="beignet"
        )

    def compose(self, other: "Rigid"):
        r = torch.einsum("...ij,...jk->...ik", self.r, other.r)
        t = self.t + torch.einsum("...ij,...j->...i", self.r, other.t)
        return Rigid(t, r)

    def inverse(self):
        return Rigid(
            -torch.einsum("...ji,...j->...i", self.r, self.t),
            self.r.transpose(-2, -1),
        )


@implements(torch.cat)
def cat(input, dim: int = 0):
    if dim < 0:
        dim = input[0].ndim + dim
    return optree.tree_map(
        lambda *x: torch.cat([*x], dim=dim), *input, namespace="beignet"
    )


@implements(torch.stack)
def stack(input, dim: int = 0):
    if dim < 0:
        dim = input[0].ndim + dim + 1
    return optree.tree_map(
        lambda *x: torch.stack([*x], dim=dim), *input, namespace="beignet"
    )


@implements(torch.unbind)
def unbind(input, dim: int = 0):
    if dim < 0:
        dim = input.ndim + dim
    return optree.tree_transpose_map(
        lambda x: torch.unbind(x, dim=dim), input, namespace="beignet"
    )


@implements(torch.unsqueeze)
def unsqueeze(input, dim: int):
    if dim < 0:
        dim = input.ndim + dim + 1
    return optree.tree_map(
        lambda x: torch.unsqueeze(x, dim=dim), input, namespace="beignet"
    )


@implements(torch.squeeze)
def squeeze(input, dim: int):
    if dim < 0:
        dim = input.ndim + dim
    return optree.tree_map(
        lambda x: torch.squeeze(x, dim=dim), input, namespace="beignet"
    )
