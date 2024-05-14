from pathlib import Path
from typing import Callable

from beignet.transforms import Transform

from .__uni_ref_dataset import _UniRefDataset


class UniRef50Dataset(_UniRefDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        index: bool = True,
        download: bool = False,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : str | Path
            Root directory where the dataset subdirectory exists or, if
            `download` is `True`, the directory where the dataset subdirectory
            will be created and the dataset downloaded.

        index : bool, optional
            If `True`, caches the sequence indicies to disk for faster
            re-initialization (default: `True`).

        download : bool, optional
            If `True`, download the dataset and to the `root` directory
            (default: `False`). If the dataset is already downloaded, it is
            not redownloaded.

        transform : Callable, optional
            A `Callable` or `Transform` that that maps a sequence to a
            transformed sequence (default: `None`).

        target_transform : Callable, optional
            A `Callable` or `Transform` that maps a target (a cluster
            identifier) to a transformed target (default: `None`).
        """
        super().__init__(
            root,
            "uniref50",
            (
                "2b05bd43c14ce0bc0591a017efa648e6",  # uniref50.fasta
                "cb28f2fc41694ccc009cbbab3e08db98",  # uniref50.fasta.gz
            ),
            index=index,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
