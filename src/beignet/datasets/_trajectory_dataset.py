import functools
from os import PathLike
from pathlib import Path
from typing import Any, Callable

try:
    from mdtraj import Trajectory
except ImportError:
    pass
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        func: Callable,
        extension: str,
        root: str | PathLike,
        transform: Callable[["Trajectory"], Any] | None = None,
        stride: int | None = None,
        **kwargs,
    ):
        self.func = functools.partial(func, **kwargs)

        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        self.transform = transform

        self.stride = stride

        self.paths = [*self.root.glob(f"*.{extension}")]

        super().__init__()

    def __getitem__(self, index: int) -> "Trajectory":
        item = self.func(self.paths[index], stride=self.stride)

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self) -> int:
        return len(self.paths)
