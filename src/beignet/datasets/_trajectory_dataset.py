from os import PathLike
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from beignet.transforms import Transform


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        transform: Callable | Transform | None = None,
        stride: int | None = None,
        *args,
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        self.transform = transform

        self.stride = stride

        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
