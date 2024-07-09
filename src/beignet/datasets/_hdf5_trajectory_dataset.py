from os import PathLike

from ._trajectory_dataset import TrajectoryDataset


class HDF5TrajectoryDataset(TrajectoryDataset):
    def __init__(self, root: str | PathLike, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
