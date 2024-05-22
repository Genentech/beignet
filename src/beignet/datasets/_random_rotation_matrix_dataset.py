from typing import Callable, Optional

from torch import Tensor
from torch.utils.data import Dataset


class RandomRotationMatrixDataset(Dataset):
    def __init__(
        self,
        size: int,
        *,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.size = size

        self.transform = transform

    def __getitem__(self, index: int) -> Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.size
