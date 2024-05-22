from typing import Callable, Generator

import torch

import beignet
from beignet.transforms import Transform

from .__random_rotation_dataset import RandomRotationDataset


class RandomEulerAngleDataset(RandomRotationDataset):
    def __init__(
        self,
        size: int,
        axes: str,
        degrees: bool | None = False,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: Generator | None = None,
        layout: torch.layout | None = torch.strided,
        pin_memory: bool | None = False,
        requires_grad: bool | None = False,
        transform: Callable | Transform | None = None,
    ) -> None:
        super().__init__(
            beignet.random_euler_angle(
                size,
                axes,
                degrees,
                generator=generator,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
            ),
            transform=transform,
        )
