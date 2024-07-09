from os import PathLike
from pathlib import Path

from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, root: str | PathLike, *args, **kwargs):
        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        super().__init__(*args, **kwargs)
