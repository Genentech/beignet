from typing import Callable, Generator

import torch

import beignet
from beignet.datasets._random_rotation_dataset import RandomRotationDataset
from beignet.transforms import Transform


class RandomRotationMatrixDataset(RandomRotationDataset):
    def __init__(
        self,
        size: int,
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
        size : int
            Output size.

        generator : torch.Generator, optional
            Psuedo-random number generator. Default, `None`.

        dtype : torch.dtype, optional
            Type of the returned tensor. Default, global default.

        layout : torch.layout, optional
            Layout of the returned tensor. Default, `torch.strided`.

        device : torch.device, optional
            Device of the returned tensor. Default, current device for the
            default tensor type.

        requires_grad : bool, optional
            Whether autograd records operations on the returned tensor.
            Default, `False`.

        pin_memory : bool, optional
            If `True`, returned tensor is allocated in pinned memory. Default,
            `False`.
        """
        super().__init__(
            beignet.random_rotation_matrix(
                size,
                generator=generator,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
            ),
            transform=transform,
        )
