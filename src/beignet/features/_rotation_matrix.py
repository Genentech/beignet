from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ._feature import Feature


class RotationMatrix(Feature):
    @classmethod
    def _wrap(cls, tensor: Tensor) -> RotationMatrix:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> RotationMatrix:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if tensor.ndim <= 1:
            raise ValueError

        if tensor.shape[-2:] == [3, 3]:
            raise ValueError

        if tensor.ndim == 2:
            tensor = torch.unsqueeze(tensor, 0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls, other: RotationMatrix, tensor: Tensor) -> RotationMatrix:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


_RotationMatrixType = Tensor | RotationMatrix
_RotationMatrixTypeJIT = Tensor

_TensorRotationMatrixType = Tensor | RotationMatrix
_TensorRotationMatrixTypeJIT = Tensor
