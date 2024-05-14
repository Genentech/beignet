from pathlib import Path
from typing import Callable

from ..transforms import Transform
from .__uni_ref_dataset import _UniRefDataset


class UniRef90Dataset(_UniRefDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        index: bool = True,
        download: bool = False,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ) -> None:
        r"""
        Parameters
        ----------
        root : str | Path
            Root directory where the dataset subdirectory exists or, if
            `download` is `True`, the directory where the dataset subdirectory
            will be created and the dataset downloaded.

        index : bool, optional
            If `True`, caches the sequence indicies to disk for faster
            re-initialization (default: `True`).

        transform : Callable, optional
            A `Callable` or `Transform` that that maps a sequence to a
            transformed sequence (default: `None`).

        target_transform : Callable, optional
            A `Callable` or `Transform` that maps a target (a cluster
            identifier) to a transformed target (default: `None`).
        """
        super().__init__(
            "http://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
            root,
            "md5:6161bad4d7506365aee882fd5ff9c833",
            index=index,
            transform=transform,
            target_transform=target_transform,
        )
