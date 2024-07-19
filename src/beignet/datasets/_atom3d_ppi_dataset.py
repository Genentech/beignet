from pathlib import Path
from typing import Callable, Tuple, Union

from pandas import DataFrame

from beignet.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DPPIDataset(ATOM3DDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform: Union[Callable, Transform, None] = None,
        target_transform: Union[Callable, Transform, None] = None,
        joint_transform_fn: Union[Callable, Transform, None] = None,
    ):
        super().__init__(
            root,
            "raw/DIPS/data",
            "https://zenodo.org/record/4911102/files/PPI-raw.tar.gz",
            "PPI",
            checksum="621977d132b39957e3480a24a30a7358",
            download=download,
        )

        self._transform_fn = transform

        self._target_transform_fn = target_transform

        self._joint_transform_fn = joint_transform_fn

    def __getitem__(self, index: int) -> Tuple[DataFrame, DataFrame]:
        item = super().__getitem__(index)

        features = DataFrame(**item["atoms_pairs"])

        target = DataFrame(**item["atoms_neighbors"])

        if self._joint_transform_fn is not None:
            features, target = self._joint_transform_fn(features, target)

        if self._transform_fn is not None:
            features = self._transform_fn(features)

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return features, target
