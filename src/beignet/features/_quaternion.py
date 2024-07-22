from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ._feature import Feature


class Quaternion(Feature):
    @classmethod
    def _wrap(cls, tensor: Tensor) -> Quaternion:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Quaternion:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if tensor.shape[-1] != 4:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls, other: Quaternion, tensor: Tensor) -> Quaternion:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


_QuaternionType = Tensor | Quaternion
_QuaternionTypeJIT = Tensor

_TensorQuaternionType = Tensor | Quaternion
_TensorQuaternionTypeJIT = Tensor
