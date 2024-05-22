from typing import Callable, Generator

import torch
from torch import Tensor
from torch.utils.data import Dataset

import beignet
from beignet.transforms import Transform


class RandomEulerAngleDataset(Dataset):
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
        super().__init__()

        self.data = beignet.random_euler_angle(
            size,
            axes,
            degrees,
            generator=generator,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )

        self.transform = transform

    def __getitem__(self, index: int) -> Tensor:
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self.data)
