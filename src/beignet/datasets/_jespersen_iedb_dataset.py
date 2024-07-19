from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from ._tdc_dataset import TDCDataset


class JespersenIEDBDataset(TDCDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        r"""
        Parameters
        ----------
        root : str | Path
            Root directory of dataset.

        download: bool
            If `True`, downloads the dataset to the root directory. If dataset
            already exists, it is not redownloaded. Default, `False`.

        transform : Callable | Transform | None
            Transforms the input.

        target_transform : Callable | Transform | None
            Transforms the target.
        """
        super().__init__(
            root=root,
            download=download,
            identifier=4165725,
            suffix="pkl",
            checksum="md5:6d5b7e005e8f8cafa117c5224698804f",
            x_keys=["X1", "X2"],
            y_keys=["Y"],
            transform=transform,
            target_transform=target_transform,
        )
