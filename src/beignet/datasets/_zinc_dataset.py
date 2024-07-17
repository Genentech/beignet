from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from ._tdc_dataset import TDCDataset


class ZINCDataset(TDCDataset):
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
            identifier=4170963,
            suffix="tsv",
            checksum="md5:95f02bd5288ac549f97d1fe7a8c3431c",
            x_keys=["smiles"],
            y_keys=[],
            transform=transform,
        )
