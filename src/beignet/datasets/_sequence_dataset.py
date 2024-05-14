from pathlib import Path

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, root: str | Path, *args, **kwargs):
        if isinstance(root, str):
            root = Path(root).resolve()

        self.root = root
