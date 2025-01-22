from pathlib import Path
from typing import Callable, Tuple, TypeVar

from beignet.transforms import Transform

from ._parquet_dataset import ParquetDataset

T = TypeVar("T")


class M320MDataset(ParquetDataset):
    """Multi-Modal Molecular (M^3) Dataset with 20M compounds.

    Reference: https://arxiv.org/abs/2412.06847

    Contains SMILES and text descriptions.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        super().__init__(
            root=root,
            path="",  # TODO: Add path
            columns=["smiles", "Description"],
            target_columns=None,
            transform=transform,
            target_transform=target_transform,
        )

        self._x = self._data[self._columns].apply(tuple, axis=1)

    def __getitem__(self, index: int) -> Tuple[T, T]:
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self._data)
