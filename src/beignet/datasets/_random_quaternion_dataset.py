from typing import Callable

from torch import Generator, Tensor
from torch.utils.data import Dataset

import beignet
from beignet.transforms import Transform


class RandomQuaternionDataset(Dataset):
    def __init__(
        self,
        size: int,
        *,
        generator: Generator = None,
        transform: Callable | Transform | None = None,
    ) -> None:
        super().__init__()

        self.size = size

        self.generator = generator

        self.transform = transform

    def __getitem__(self, _index: int) -> Tensor:
        return beignet.random_quaternion(1, generator=self.generator)

    def __len__(self) -> int:
        return self.size
