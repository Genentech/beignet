from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pandas

from beignet.transforms import Transform

from ._data_frame_dataset import DataFrameDataset


class ParquetDataset(DataFrameDataset):
    def __init__(
        self,
        root: Union[str, Path],
        path: Union[str, Path],
        *,
        columns: Optional[Sequence[str]],
        target_columns: Optional[Sequence[str]],
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
        **kwargs,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param columns: x features of the dataset. items in the dataset are
            of the form ((columns), (target_columns)).

        :param target_columns: y features of the dataset. items in the dataset
            are of the form ((columns), (target_columns)).

        :param transform_fn: A ``Callable`` or ``Transform`` that maps data to
            transformed data (default: ``None``).

        :param target_transform_fn: ``Callable`` or ``Transform`` that maps a
            target to a transformed target (default: ``None``).
        """
        super().__init__(
            root,
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )

        self._path = path

        self._columns = columns

        self._target_columns = target_columns

        self._data = pandas.read_parquet(self._path, **kwargs)
