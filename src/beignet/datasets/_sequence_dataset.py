from pathlib import Path
from typing import Any, Union

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, root: Union[str, Path], *args, **kwargs) -> None:
        if isinstance(root, str):
            root = Path(root).resolve()

        self.root = root

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        subject = self.__class__.__name__

        records = [f"SEQUENCES: {self.__len__()}"]

        if self.root is not None:
            records = [*records, f"PATH: {self.root}"]

        messages = []

        for message in records:
            messages = [*messages, f"    {message}"]

        return "\n".join([subject, *messages])
