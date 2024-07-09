from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset

from beignet.transforms import Transform


class RandomRotationDataset(Dataset):
    def __init__(
        self,
        data: Tensor,
        *,
        transform: Callable | Transform | None = None,
    ):
        super().__init__()

        self.data = data

        self.transform = transform

    def __getitem__(self, index: int) -> Tensor:
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self.data)
