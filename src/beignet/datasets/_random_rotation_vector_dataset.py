from typing import Callable, Generator

import torch
from torch.utils.data import Dataset

import beignet
from beignet.transforms import Transform


class RandomRotationVectorDataset(Dataset):
    def __init__(
        self,
        size: int,
        degrees: bool = False,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: Generator | None = None,
        layout: torch.layout | None = torch.strided,
        pin_memory: bool | None = False,
        requires_grad: bool | None = False,
        transform: Callable | Transform | None = None,
    ):
        r"""
        Parameters
        ----------
        """
        super().__init__(
            beignet.random_rotation_vector(
                size,
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
