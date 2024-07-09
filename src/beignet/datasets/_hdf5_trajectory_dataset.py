from os import PathLike
from typing import Any, Callable

import mdtraj
from mdtraj import Trajectory

from ._trajectory_dataset import TrajectoryDataset


class HDF5TrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        root: str | PathLike,
        transform: Callable[[Trajectory, ...], Any] | None = None,
        stride: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(root, transform, stride, *args, **kwargs)

        self.paths = self.root.glob("*.hdf5")

    def __getitem__(self, index: int) -> Trajectory:
        item = mdtraj.load_hdf5(self.paths[index], stride=self.stride)

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.paths)
