from os import PathLike
from typing import Callable

from beignet.transforms import Transform

from ._uniprot_dataset import UniProtDataset


class SwissProtDataset(UniProtDataset):
    def __init__(
        self,
        root: str | PathLike | None = None,
        *,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        """
        Parameters
        ----------
        root : str | PathLike, optional
            Root directory where the dataset subdirectory exists or, if
            `download` is `True`, the directory where the dataset subdirectory
            will be created and the dataset downloaded.

        transform : Callable, optional
            A `Callable` or `Transform` that that maps a sequence to a
            transformed sequence (default: `None`).

        target_transform : Callable, optional
            A `Callable` or `Transform` that maps a target to a transformed
            target (default: `None`).
        """
        super().__init__(
            "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
            root,
            "md5:0766df3e5785fc5f1cfc496aa89e86ad",
            transform=transform,
            target_transform=target_transform,
        )
