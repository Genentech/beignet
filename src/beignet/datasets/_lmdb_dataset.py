import json
from gzip import GzipFile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict

import lmdb
from torch.utils.data import Dataset

from beignet.transforms import Transform


class LMDBDataset(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        *,
        lock: bool = False,
        max_readers: int = 1,
        meminit: bool = True,
        readahead: bool = True,
        readonly: bool = True,
        transform: Callable | Transform | None = None,
    ):
        super().__init__()

        self.root = root

        self.transform = transform

        if isinstance(self.root, str):
            self.root = Path(self.root).resolve()

        self.data = lmdb.open(
            str(self.root),
            lock=lock,
            max_readers=max_readers,
            meminit=meminit,
            readahead=readahead,
            readonly=readonly,
        )

        with self.data.begin(write=False) as transaction:
            self.size = int(transaction.get(b"num_examples"))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < self.size:
            raise IndexError(index)

        with self.data.begin(write=False) as transaction:
            with GzipFile(
                fileobj=BytesIO(transaction.get(str(index).encode())),
                mode="rb",
            ) as descriptor:
                item = json.loads(descriptor.read())

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self) -> int:
        return self.size
