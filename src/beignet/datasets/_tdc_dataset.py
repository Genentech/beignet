from pathlib import Path
from typing import Callable, List, Tuple, TypeVar

import pandas
import pooch
from torch.utils.data import Dataset

from beignet.transforms import Transform

T = TypeVar("T")


class TDCDataset(Dataset):
    _x: List[T]
    _y: List[T]

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        *,
        identifier: int,
        suffix: str,
        checksum: str,
        x_keys: List[str],
        y_keys: List[str] | None = None,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        if download:
            pooch.retrieve(
                f"https://dataverse.harvard.edu/api/access/datafile/{identifier}",
                fname=f"{self.__class__.__name__}.{suffix}",
                known_hash=checksum,
                path=root / self.__class__.__name__,
                progressbar=True,
            )

        path = root / self.__class__.__name__ / f"{self.__class__.__name__}.{suffix}"

        match path.suffix:
            case ".csv":
                self._data = pandas.read_csv(path, sep=None)
            case ".pkl":
                self._data = pandas.read_pickle(path)
            case ".tab" | ".tsv":
                self._data = pandas.read_csv(path, sep="\t")
            case _:
                raise ValueError

        self._x_keys = x_keys
        self._y_keys = y_keys

        self.transform = transform
        self.target_transform = target_transform

        self._x = self._data[self._x_keys].apply(tuple, axis=1)

        if self._y_keys is not None:
            self._y = self._data[self._y_keys].apply(tuple, axis=1)

    def __getitem__(self, index: int) -> Tuple[T, T]:
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        if self._y_keys is None:
            return x

        y = self._y[index]

        if len(y) == 1:
            y = y[0]

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self) -> int:
        return len(self._data)
