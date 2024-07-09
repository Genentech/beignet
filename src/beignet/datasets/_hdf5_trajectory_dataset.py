from os import PathLike
from typing import Any, Callable

import mdtraj
from mdtraj import Trajectory

from ._trajectory_dataset import TrajectoryDataset


class HDF5TrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        root: str | PathLike,
        transform: Callable[[Trajectory], Any] | None = None,
        stride: int | None = None,
        **kwargs,
    ):
        super().__init__(
            func=mdtraj.load_hdf5,
            extension="hdf5",
            root=root,
            transform=transform,
            stride=stride,
            **kwargs,
        )
