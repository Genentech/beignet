from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from ._tdc_dataset import TDCDataset


class MOSESDataset(TDCDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform: Callable | Transform | None = None,
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
        """
        super().__init__(
            root=root,
            download=download,
            identifier=4170962,
            suffix="moses.tab",
            checksum="md5:b684443540f42cbdebb63ad090a1b4b3",
            x_keys=["smiles"],
            transform=transform,
        )
