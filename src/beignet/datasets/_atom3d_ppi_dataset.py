from os import PathLike
from typing import Callable

from pandas import DataFrame

from beignet.transforms import Transform

from .__atom3d_dataset import ATOM3DDataset


class ATOM3DPPIDataset(ATOM3DDataset):
    def __init__(
        self,
        root: str | PathLike,
        *,
        download: bool = False,
        joint_transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
        transform: Callable | Transform | None = None,
    ):
        super().__init__(
            root,
            "raw/DIPS/data",
            "https://zenodo.org/record/4911102/files/PPI-raw.tar.gz",
            "PPI",
            checksum="621977d132b39957e3480a24a30a7358",
            download=download,
        )

        self.joint_transform = joint_transform

        self.target_transform = target_transform

        self.transform = transform

    def __getitem__(self, index: int) -> (DataFrame, DataFrame):
        item = super().__getitem__(index)

        input = DataFrame(**item["atoms_pairs"])

        target = DataFrame(**item["atoms_neighbors"])

        if self.joint_transform is not None:
            input, target = self.joint_transform(input, target)

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target
