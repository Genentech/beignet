import json
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Union

from torch.utils.data import Dataset

from beignet.transforms import Transform


class LMDBDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        lock: bool = False,
        max_readers: int = 1,
        meminit: bool = True,
        readahead: bool = True,
        readonly: bool = True,
        transform: Union[Callable, Transform, None] = None,
    ):
        super().__init__()

        try:
            import lmdb
        except ImportError as error:
            raise ImportError(
                """
                LMDB datasets require the `lmdb` dependency:

                    $ pip install "beignet[lmdb]"
                """
            ) from error

        self._root = root

        self._transform_fn = transform

        if isinstance(self._root, str):
            self._root = Path(self._root).resolve()

        self._data = lmdb.open(
            str(self._root),
            lock=lock,
            max_readers=max_readers,
            meminit=meminit,
            readahead=readahead,
            readonly=readonly,
        )

        with self._data.begin(write=False) as transaction:
            self._size = int(transaction.get(b"num_examples"))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < self._size:
            raise IndexError(index)

        with self._data.begin(write=False) as transaction:
            with GzipFile(
                fileobj=BytesIO(transaction.get(str(index).encode())),
                mode="rb",
            ) as descriptor:
                item = json.loads(descriptor.read())

        if self._transform_fn:
            item = self._transform_fn(item)

        return item

    def __len__(self) -> int:
        return self._size
