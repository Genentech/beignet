from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ._feature import Feature


class EulerAngle(Feature):
    @classmethod
    def _wrap(cls, tensor: Tensor) -> EulerAngle:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> EulerAngle:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if tensor.ndim == 0:
            raise ValueError

        if tensor.shape[-1] != 3:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls, other: EulerAngle, tensor: Tensor) -> EulerAngle:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


_EulerAngleType = Tensor | EulerAngle
_EulerAngleTypeJIT = Tensor

_TensorEulerAngleType = Tensor | EulerAngle
_TensorEulerAngleTypeJIT = Tensor
