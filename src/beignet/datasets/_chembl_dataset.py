from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from ._tdc_dataset import TDCDataset


class ChEMBLDataset(TDCDataset):
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
            identifier=4170965,
            suffix="csv",
            checksum="md5:b0cedfe468c6331487f7084236944bac",
            x_keys=["smiles"],
            y_keys=None,
            transform=transform,
        )
