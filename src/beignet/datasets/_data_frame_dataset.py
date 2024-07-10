from pathlib import Path
from typing import Callable, TypeVar, Union

from pandas import DataFrame
from torch.utils.data import Dataset

from beignet.transforms import Transform

T = TypeVar("T")


class DataFrameDataset(Dataset):
    _data: DataFrame

    def __init__(
        self,
        root: Union[str, Path],
        *,
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param transform_fn: A ``Callable`` or ``Transform`` that maps data to
            transformed data (default: ``None``).

        :param target_transform_fn: ``Callable`` or ``Transform`` that maps a
            target to a transformed target (default: ``None``).
        """
        if isinstance(root, str):
            root = Path(root).resolve()

        self._root = root

        self._transform_fn = transform_fn

        self._target_transform_fn = target_transform_fn

    def __getitem__(self, index: int) -> T:
        return self._data.iloc[index]

    def __len__(self) -> int:
        return len(self._data)
