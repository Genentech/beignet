from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from ._tdc_dataset import TDCDataset


class TherapeuticAntibodyProfilerDataset(TDCDataset):
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
            identifier=4167113,
            suffix="tsv",
            checksum="md5:0a1b07fe1bdc9f67636f72878097841e",
            x_keys=["X"],
            y_keys=[
                "CDR_Length",
                "PNC",
                "PPC",
                "PSH",
                "SFvCSP",
            ],
            transform=transform,
            target_transform=target_transform,
        )
