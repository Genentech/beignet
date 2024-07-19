from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from pandas import DataFrame
from torch import Tensor

from beignet.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DMSPDataset(ATOM3DDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform: Union[Callable, Transform, None] = None,
        target_transform: Union[Callable, Transform, None] = None,
    ):
        super().__init__(
            root,
            "raw/MSP/data",
            "https://zenodo.org/record/4962515/files/MSP-raw.tar.gz",
            "MSP",
            checksum="77aeb79cfc80bd51cdfb2aa321bf6128",
            download=download,
        )

        self._transform_fn = transform

        self._target_transform_fn = target_transform

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Tuple[DataFrame, DataFrame], Tensor]:
        item = super().__getitem__(index)

        structure = DataFrame(**item["original_atoms"])

        mutant = DataFrame(**item["mutated_atoms"])

        if self._transform_fn is not None:
            structure, mutant = self._transform_fn(structure, mutant)

        target = torch.tensor(int(item["label"]))

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return (structure, mutant), target
