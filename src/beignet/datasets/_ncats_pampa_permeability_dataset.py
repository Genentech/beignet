from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from ._tdc_dataset import TDCDataset


class NCATSPAMPAPermeabilityDataset(TDCDataset):
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
            identifier=6695857,
            suffix="approved_pampa_ncats.tab",
            checksum="2b1f7275a768da97cc01c0187958e2350f40beddc5f20ade0eb6317cdaa2262c",
            x_keys=["Drug"],
            y_keys=["Y"],
            transform=transform,
            target_transform=target_transform,
        )
