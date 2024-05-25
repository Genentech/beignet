from typing import Callable, Sequence, TypeVar

import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from beignet.transforms import Transform

T = TypeVar("T")


class DataFrameDataset(Dataset):
    data: DataFrame

    def __init__(
        self,
        data: DataFrame,
        *,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
        columns: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
    ):
        self.data = data

        self.transform = transform

        self.target_transform = target_transform

        self.columns = columns

        self.target_columns = target_columns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T | (T, T):
        item = self.data.iloc[index]

        if len(self.columns) > 1:
            input = tuple(item[column] for column in self.columns)
        else:
            input = item[self.columns[0]]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_columns is None:
            return input

        if len(self.target_columns) > 1:
            target = tuple(item[column] for column in self.target_columns)
        else:
            target = item[self.target_columns[0]]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if len(self.target_columns) > 1:
            if not all(isinstance(y, Tensor) for y in target):
                target = tuple(torch.as_tensor(y) for y in target)
            elif not isinstance(target, Tensor):
                target = torch.as_tensor(target)
        elif not isinstance(target, Tensor):
            target = torch.as_tensor(target)

        return input, target
