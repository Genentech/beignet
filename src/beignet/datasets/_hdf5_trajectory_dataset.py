from os import PathLike
from typing import Any, Callable

try:
    from mdtraj import Trajectory
except ImportError:
    pass
import beignet

from ._trajectory_dataset import TrajectoryDataset


class HDF5TrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        root: str | PathLike,
        transform: Callable[["Trajectory"], Any] | None = None,
        stride: int | None = None,
        **kwargs,
    ):
        if beignet.optional_dependencies(["mdtraj"], ["mdtraj"]):
            import mdtraj

            super().__init__(
                func=mdtraj.load_hdf5,
                extension="hdf5",
                root=root,
                transform=transform,
                stride=stride,
                **kwargs,
            )
