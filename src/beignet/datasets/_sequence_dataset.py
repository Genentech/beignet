from os import PathLike
from pathlib import Path

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, root: str | PathLike, *args, **kwargs):
        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()
