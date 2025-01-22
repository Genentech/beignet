from os import PathLike
from typing import Any, Callable

import beignet

try:
    from mdtraj import Trajectory
except ImportError:
    pass

from ._trajectory_dataset import TrajectoryDataset


class PDBTrajectoryDataset(TrajectoryDataset):
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
                func=mdtraj.load_pdb,
                extension="pdb",
                root=root,
                transform=transform,
                stride=stride,
                **kwargs,
            )
