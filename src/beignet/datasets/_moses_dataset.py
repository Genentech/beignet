from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from .__tdc_dataset import _TDCDataset


class MOSESDataset(_TDCDataset):
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
            checksum="0771d24e56f1b281ec4b0cdb1c85bab2b74ee9f34c7424d2b4432aa4a593d4c2",
            x_keys=["smiles"],
            transform=transform,
        )
